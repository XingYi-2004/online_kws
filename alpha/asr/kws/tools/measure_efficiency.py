#!/usr/bin/env python3
"""FLOPs / SOPs / Energy 测量脚本（真实测试集版本）。

用法: python scripts/measure_efficiency.py

读取 measure_config.yaml，对每个模型：
  1. 加载预训练权重
  2. 加载对应数据集的真实测试数据
  3. 每条样本用自己的实际帧数 forward
  4. 统计 FLOPs / SOPs，全量跑完取平均
  5. 计算 Energy 并打印汇总表格
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# 框架根（tools → kws → asr → alpha → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
np.int = int
import torch
import torchaudio
import yaml
from spikingjelly.activation_based import functional, neuron

# ============================================================
# 能耗常量 (45nm CMOS)
# ============================================================
E_MAC = 4.6e-12  # J per MAC (浮点乘加)
E_AC  = 0.9e-12  # J per AC  (脉冲累加)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 模型构建
# ============================================================

def _build_tdnnlstm(entry, ckpt, device):
    from alpha.asr.model.tdnn_lstm import TdnnLstm
    model = TdnnLstm(
        num_features=80,
        num_classes=entry["num_class"],
        subsampling_factor=entry.get("subsampling_factor", 3),
    )
    model.load_state_dict(ckpt["_sd"])
    return model


def _build_alphasnn(entry, ckpt, device):
    import re

    from alpha.asr.model.multitau_lif_commander_kws import MultiTauLIFCommanderKWS

    cfg_path = PROJECT_ROOT / entry["config"]
    yaml_cfg = _load_yaml(cfg_path)

    m = yaml_cfg.get("model", {})
    if "param" in m:
        model_cfg = dict(m["param"])
    else:
        skip_keys = {"name", "mask", "debug"}
        model_cfg = {k: v for k, v in m.items() if k not in skip_keys and not k.startswith("_")}

    sd = ckpt["_sd"]
    for k in ("model.frontend.pointwise.weight", "model.frontend.linear.weight"):
        if k in sd:
            model_cfg["n_hidden_neurons"] = sd[k].shape[0]
            break
    block_ids = set()
    for k in sd:
        m = re.match(r"model\.blocks\.(\d+)\.", k)
        if m:
            block_ids.add(int(m.group(1)))
    if block_ids:
        model_cfg["depths"] = max(block_ids) + 1
    for k in ("model.classifier.weight", "model.head.weight"):
        if k in sd:
            model_cfg["n_outputs"] = sd[k].shape[0]
            break
    for k in ("model.frontend.pointwise.weight", "model.frontend.linear.weight"):
        if k in sd:
            model_cfg["n_inputs"] = sd[k].shape[1]
            break

    model = MultiTauLIFCommanderKWS(model_cfg)
    # model.* / backbone.* 前缀一律剥掉，落到继承模型的裸键（frontend/blocks/classifier）
    stripped = {}
    for k, v in sd.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        elif k.startswith("backbone."):
            stripped[k[len("backbone."):]] = v
        else:
            stripped[k] = v
    model.load_state_dict(stripped)
    return model


BUILDERS = {
    "tdnnlstm":     _build_tdnnlstm,
    "alphasnn":     _build_alphasnn,
}


def build_model(entry, device):
    ckpt_path = PROJECT_ROOT / entry["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # 统一提取 state_dict
    if isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    elif isinstance(raw, dict) and any(k.startswith("module.") for k in list(raw.keys())[:5]):
        sd = {k.replace("module.", "", 1): v for k, v in raw.items()}
    else:
        sd = raw  # 裸 state_dict
    ckpt = dict(raw) if isinstance(raw, dict) else {}
    ckpt["_sd"] = sd
    builder = BUILDERS[entry["type"]]
    model = builder(entry, ckpt, device)
    model.to(device)
    model.eval()
    return model


# ============================================================
# 测试数据加载
# ============================================================

def _get_sample_rate(test_cfg):
    """返回测试数据的采样率（默认 16000）。"""
    return int(test_cfg.get("sample_rate", 16000))


def load_test_data(test_cfg):
    """加载测试集，返回 list of (T, 80) tensors。"""
    fmt = test_cfg["format"]

    if fmt == "memmap":
        sys.path.insert(0, str(PROJECT_ROOT / "offline"))
        from memmap_dataset import MemmapKWSDataset
        ds = MemmapKWSDataset(
            memmap_path=str(PROJECT_ROOT / test_cfg["feature_file"]),
            index_path=str(PROJECT_ROOT / test_cfg["index_file"]),
        )
        samples = []
        for i in range(len(ds)):
            feats, _ = ds[i]  # (T, 80)
            samples.append(feats)
        return samples

    elif fmt == "lhotse":
        from lhotse import load_manifest
        manifest = str(PROJECT_ROOT / test_cfg["manifest"])
        cuts = load_manifest(manifest)
        samples = []
        for cut in cuts:
            feats = cut.load_features()  # tensor (T, 80), or ndarray
            if not isinstance(feats, torch.Tensor):
                feats = torch.from_numpy(feats)
            samples.append(feats.float())
        return samples

    elif fmt == "data_list":
        import json
        data_list_path = str(PROJECT_ROOT / test_cfg["manifest"])
        n_mels = test_cfg.get("n_mels", 80)
        sample_rate = _get_sample_rate(test_cfg)

        from torchaudio.compliance import kaldi
        samples = []
        with open(data_list_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                waveform, sr = torchaudio.load(obj["wav"])
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                fbank = kaldi.fbank(
                    waveform * 32768.0,
                    num_mel_bins=n_mels,
                    sample_frequency=sample_rate,
                    frame_length=25.0,
                    frame_shift=10.0,
                    window_type='hamming',
                    use_energy=False,
                )  # (T, n_mels)
                samples.append(fbank.float())
        return samples

    elif fmt == "gsc":
        from torchaudio.datasets import SPEECHCOMMANDS
        from torchaudio import transforms

        root = str(PROJECT_ROOT / test_cfg.get("root", "gsc"))
        n_mels = test_cfg.get("n_mels", 80)

        # FBank 变换（与训练一致的参数）
        fbank = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=50,
            f_max=8000 if n_mels == 80 else 14000,
        )

        ds = SPEECHCOMMANDS(root=root, subset="testing", download=False)
        samples = []
        for waveform, sample_rate, label, *_ in ds:
            if sample_rate != 16000:
                waveform = transforms.Resample(sample_rate, 16000)(waveform)
            feats = fbank(waveform).squeeze(0).T  # (freq, time) → (T, n_mels)
            samples.append(feats.float())
        return samples

    else:
        raise ValueError(f"Unknown test_data format: {fmt}")


# ============================================================
# 单样本测量
# ============================================================

def _format_input(feats, model_type, device):
    """将 (T, 80) 转为模型需要的输入格式 [1, ...] 并移到 device。"""
    x = feats.unsqueeze(0).to(device)  # [1, T, 80]
    return x.transpose(1, 2)  # [B, C, T] for tdnnlstm/alphasnn


def _compute_fanouts(model):
    """遍历模型，为每个脉冲层计算其下游突触 fan-out（SpikCommander 附录E Eq.18 口径）。

    SOP^l = fr^(l-1) × FLOP^l，等价于：脉冲数 × fan-out。
      - 脉冲层 → 下一个 Conv1d/Linear：存在突触，fan_out = 每个输入单元的连接数
        (Conv1d: (out_channels//groups)×kernel_size；Linear: out_features)
      - 脉冲层 → 下一个脉冲层（如残差进 LIF 膜电位）：膜电位积分，无突触，fan_out=0
    返回 {module_id: fan_out} 字典。
    """
    # 按 forward 顺序收集所有模块
    modules_in_order = list(model.named_modules())
    fan_out_map = {}

    for i, (name, module) in enumerate(modules_in_order):
        # 只处理脉冲层
        is_spike = isinstance(module, (neuron.LIFNode, neuron.ParametricLIFNode, neuron.IFNode))
        is_spike_rnn = type(module).__name__ in ("SpikingVanillaRNN", "SpikingLSTM")
        if not is_spike and not is_spike_rnn:
            continue

        # 向后找第一个"计算层"（Conv1d / Linear）；途中遇到脉冲层则停止
        # （脉冲→脉冲经膜电位积分传递，不产生突触操作）
        fan_out = 0
        for j in range(i + 1, len(modules_in_order)):
            next_mod = modules_in_order[j][1]
            if isinstance(next_mod, torch.nn.Conv1d):
                fan_out = (next_mod.out_channels // next_mod.groups) * next_mod.kernel_size[0]
                break
            elif isinstance(next_mod, torch.nn.Linear):
                fan_out = next_mod.out_features
                break
            elif _is_spike_module(next_mod):
                break
            # 跳过 BN / Dropout / 池化等非计算层，继续向后找
        fan_out_map[id(module)] = fan_out

    return fan_out_map


def _make_sops_hooks(model):
    """注册 SOP 钩子（含 fan_out），返回 hooks 列表和 (spike_counts, fan_out_map)。

    论文式口径：每个 LIF 的输出脉冲数 × 下游计算层 fan-out 计为 SOPs。
    """
    fan_out_map = _compute_fanouts(model)
    spike_counts = []
    hooks = []

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor):
            fan_out = fan_out_map.get(id(module), 0)
            raw_spikes = (output > 0).sum().item()
            spike_counts.append(raw_spikes * fan_out)

    for _, module in model.named_modules():
        if isinstance(module, (neuron.LIFNode, neuron.ParametricLIFNode, neuron.IFNode)):
            hooks.append(module.register_forward_hook(_hook))
        elif type(module).__name__ in ("SpikingVanillaRNN", "SpikingLSTM"):
            hooks.append(module.register_forward_hook(_hook))

    return hooks, spike_counts


def _is_spike_module(module):
    """判断模块是否为脉冲层（LIF / IF / 脉冲 RNN）。"""
    return (
        isinstance(module, (neuron.LIFNode, neuron.ParametricLIFNode, neuron.IFNode))
        or type(module).__name__ in ("SpikingVanillaRNN", "SpikingLSTM")
    )


def _find_first_spike_idx(model):
    """返回第一个脉冲层在 named_modules 顺序中的下标；无脉冲层（纯 ANN）返回 None。"""
    for i, (_, m) in enumerate(model.named_modules()):
        if _is_spike_module(m):
            return i
    return None


def _make_frontend_macs_hooks(model):
    """统计脉冲编码之前的真实值层 MAC（SpikCommander 附录E Eq.20 口径）。

    Eq.20: E = E_MAC × FLOP_Conv^1 + E_AC × Σ SOP。
    即：只有第一个脉冲层之前、接收连续输入（如 FBank/Mel）的层计入
    FLOPs/MAC；脉冲编码之后的所有 Conv/Linear（含分类头）一律按
    SOP 计（级联线性层可融合，仍视为 spike-form 输入的 AC 操作）。
    公式与 thop 一致：
      Conv1d: y.nelement() × (in_channels // groups × kernel_size)
      Linear: y.nelement() × in_features
    返回 (hooks, macs_map)；无脉冲层（纯 ANN）时 hooks 为空。
    """
    first_spike = _find_first_spike_idx(model)
    macs_map = {}
    hooks = []
    if first_spike is None:
        return hooks, macs_map

    modules_in_order = list(model.named_modules())
    for i in range(first_spike):
        _, m = modules_in_order[i]
        if isinstance(m, torch.nn.Conv1d):
            hooks.append(_conv1d_macs_hook(m, macs_map))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(_linear_macs_hook(m, macs_map))
    return hooks, macs_map


def _conv1d_macs_hook(m, macs_map):
    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        macs_map[id(module)] = int(
            y.nelement() * (module.in_channels // module.groups * module.kernel_size[0])
        )
    return m.register_forward_hook(hook)


def _linear_macs_hook(m, macs_map):
    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        macs_map[id(module)] = int(y.nelement() * module.in_features)
    return m.register_forward_hook(hook)


def measure_sample(model, x, model_type):
    """对一条样本测 FLOPs + SOPs（SpikCommander 附录E Eq.18/20 口径），返回 (flops, sops)。

    FLOPs/MAC = 第一个脉冲层之前的真实值层（Eq.20 的 FLOP_Conv^1）；
    SOPs = 各脉冲层 spike 数 × 下游突触 fan-out（Eq.18: fr × FLOP），
    脉冲→脉冲为膜电位积分不计 SOP；分类头等脉冲后线性层全部计 SOP。
    纯 ANN 用 thop 总量（Eq.17: E_MAC × Σ MAC）。
    """
    from thop import profile

    functional.reset_net(model)

    # 注册 SOP 钩子 + frontend MAC 钩子
    hooks, spike_counts = _make_sops_hooks(model)
    frontend_hooks, frontend_macs = _make_frontend_macs_hooks(model)

    # thop 内部会 forward，同时触发钩子
    total_flops, _ = profile(model, inputs=(x,), verbose=False)

    # SNN：FLOPs=脉冲编码前真实值层；纯 ANN：thop 总量
    if frontend_hooks:
        flops = float(sum(frontend_macs.values()))
    else:
        flops = total_flops

    sops = sum(spike_counts) if model_type != "tdnnlstm" else 0

    # 清理
    for h in hooks + frontend_hooks:
        h.remove()
    functional.reset_net(model)

    return flops, sops



def measure_on_testset(model, test_samples, model_type, device, max_samples=None):
    """遍历测试集，取平均 FLOPs / SOPs。"""
    import random
    all_flops, all_sops = [], []

    # 采样
    if max_samples and max_samples < len(test_samples):
        indices = random.sample(range(len(test_samples)), max_samples)
    else:
        indices = range(len(test_samples))

    for i in indices:
        x = _format_input(test_samples[i], model_type, device)
        try:
            flops, sops = measure_sample(model, x, model_type)
            all_flops.append(flops)
            all_sops.append(sops)
        except Exception as e:
            print(f"\n    ! sample {i} failed: {e}")
            continue

    return (
        sum(all_flops) / len(all_flops) if all_flops else 0,
        sum(all_sops) / len(all_sops) if all_sops else 0,
    )


# ============================================================
# 格式化
# ============================================================

def _fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"


def estimate_energy(flops, sops):
    return (flops * E_MAC + sops * E_AC) * 1e3  # mJ


# ============================================================
# 主流程
# ============================================================

def _test_data_key(test_cfg):
    """返回测试数据的唯一标识，相同标识=可复用缓存。"""
    return tuple(sorted(test_cfg.items()))


class _Tee:
    """同时输出到终端和日志文件。"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()


def main():
    # 日志输出到 scripts/ 目录，带时间戳
    log_dir = PROJECT_ROOT / "scripts"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"measure_{timestamp}.log"
    tee = _Tee(str(log_path))
    sys.stdout = tee

    cfg_path = PROJECT_ROOT / "scripts" / "measure_config.yaml"
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)

    cfg = _load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"Log: {log_path}\n")

    # 提前加载所有测试数据（在导入 AlphaSNN 之前，避免编码冲突）
    print("Loading test datasets ...")
    test_data_cache = {}
    for entry in (cfg.get("models") or []):
        td = entry.get("test_data")
        if td is None:
            continue
        key = _test_data_key(td)
        if key not in test_data_cache:
            try:
                samples = load_test_data(td)
                test_data_cache[key] = samples
                print(f"  {td['format']}: {len(samples)} samples")
            except Exception as e:
                print(f"  {td['format']}: FAILED — {e}")
                test_data_cache[key] = None
    print()

    results = []

    for entry in (cfg.get("models") or []):
        name = entry["name"]
        model_type = entry["type"]

        if entry.get("checkpoint") is None:
            print(f"  -  {name}: checkpoint=null, skipped")
            continue

        td = entry.get("test_data")
        if td is None:
            print(f"  -  {name}: test_data 未配置, skipped")
            continue

        test_samples = test_data_cache.get(_test_data_key(td))
        if test_samples is None:
            print(f"  -  {name}: test data failed to load, skipped")
            continue

        print(f"  Measuring {name} ...", flush=True)

        # 1. 构建模型
        try:
            model = build_model(entry, device)
        except Exception as e:
            import io
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            print(f"  FAILED (build): {e}")
            print(buf.getvalue())
            continue

        params = sum(p.numel() for p in model.parameters())

        # 2. 测量取平均（采样 max_samples 条）
        max_samples = cfg.get("input", {}).get("max_samples", None)
        try:
            avg_flops, avg_sops = measure_on_testset(
                model, test_samples, model_type, device, max_samples=max_samples
            )
        except Exception as e:
            print(f"  FAILED (measure): {e}")
            continue

        avg_energy = estimate_energy(avg_flops, avg_sops)

        results.append({
            "name": name,
            "params": params,
            "flops": avg_flops,
            "sops": avg_sops,
            "energy": avg_energy,
            "n_samples": len(test_samples),
            "sample_rate": _get_sample_rate(td),
            "input_dim": int(test_samples[0].shape[1]) if test_samples else 0,
        })

        n = len(test_samples)
        print(f"\r  {name}: {n} samples → "
              f"FLOPs={_fmt(avg_flops)}, SOPs={_fmt(avg_sops)}, Energy={avg_energy:.3f}mJ")

    # ============================================================
    # 汇总表格
    # ============================================================
    if not results:
        print("\nNo results.")
        return

    print("\n" + "=" * 112)
    print(f"{'Model':<36} {'Params':>8} {'FLOPs':>10} {'SOPs':>10} {'Energy':>10} {'Samples':>8} {'SR':>6} {'Dim':>5}")
    print("-" * 112)
    for r in results:
        sr = f"{r['sample_rate']//1000}kHz"
        print(f"{r['name']:<36} {_fmt(r['params']):>8} {_fmt(r['flops']):>10} "
              f"{_fmt(r['sops']):>10} {r['energy']:>7.3f} mJ {r['n_samples']:>8} "
              f"{sr:>6} {r['input_dim']:>5}")
    print("=" * 112)

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
