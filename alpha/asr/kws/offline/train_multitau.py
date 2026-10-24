"""离线训练脚本（重建版）。

原版在 2026-10-08 清理中误删；本脚本依据训练日志与 checkpoint 键结构重建，
超参与 KD-MT-08（干净测试 98.04%）一致：lr=2e-3、warmup 30 + cosine 270、
EMA 0.9995、KD 教师 TdnnLstm ANN（α=0.7，T=10.0）、seed=312。
参数键（subsample.* + model.*）与历史 checkpoint 兼容，在线管线可直接加载。
评测细节与原版存在口径差异，详见 offline/README.md。

用法：
  cd AlphaSNN && conda run -n snn python alpha/asr/kws/offline/train_multitau.py \
      --conf alpha/asr/kws/conf/offline_kd.yaml
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

if not hasattr(np, "int"):
    np.int = int

# AlphaSNN 框架根（offline → kws → asr → alpha → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from lhotse.dataset import DynamicBucketingSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from tqdm.auto import tqdm

from alpha.asr.model.multitau_lif_commander.multitau_lif_commander import (
    MultiTauLIFCommander,
)
from alpha.asr.loss.spiking_loss import SpikingCELoss
from alpha.asr.kws.utils.data import _make_dataset, load_cuts
from alpha.asr.kws.utils.train_utils import extract_labels

# 模型封装：参数键 subsample.* + model.*（与历史 checkpoint 兼容）
class AlphaSNNForKWS(nn.Module):
    """AvgPool 下采样 + MultiTauLIFCommander，前向流程与在线封装一致。"""

    def __init__(self, config: dict):
        super().__init__()
        ss = int(config.get("subsampling_factor", 2))
        self.subsample_factor = ss
        self.subsample = nn.AvgPool1d(ss, stride=ss) if ss > 1 else nn.Identity()
        self.model = MultiTauLIFCommander(config)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, F, T]
        x = self.subsample(x).transpose(1, 2)  # [B, T', F]
        mask = None
        if lengths is not None:
            num_frames = x.shape[1]
            adj = torch.clamp((lengths + self.subsample_factor - 1) // self.subsample_factor,
                              min=1, max=num_frames)
            mask = torch.arange(num_frames, device=x.device).unsqueeze(0) < adj.unsqueeze(1)
        return self.model(x, mask)


# EMA：权重的指数滑动平均，需装在同结构模型副本里才能前向评测/保存
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd, sd = self.ema_model.state_dict(), model.state_dict()
        for k in msd:
            if msd[k].dtype.is_floating_point:
                msd[k].mul_(self.decay).add_(sd[k], alpha=1.0 - self.decay)
            else:
                msd[k].copy_(sd[k])


def make_loader(manifest: Path, max_duration: float, shuffle: bool = False,
                num_buckets: int = 30, num_workers: int = 8, persistent: bool = False):
    cuts = load_cuts(manifest.parent, manifest.name)
    sampler = DynamicBucketingSampler(cuts, max_duration=max_duration,
                                      shuffle=shuffle, num_buckets=num_buckets,
                                      drop_last=False)
    loader = DataLoader(_make_dataset(None, False), sampler=sampler, batch_size=None,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=persistent)
    try:    # 动态采样器无 len()，用总时长估算批数（仅供进度条显示，允许有误差）
        loader.est_batches = int(sum(c.duration for c in cuts) / max_duration) + 1
    except Exception:
        loader.est_batches = 0
    return loader


# SpecAugment：训练时对 FBank 特征做时频遮罩（评测不增强）
def spec_augment(features: torch.Tensor, time_mask: int, freq_mask: int,
                 num_time_masks: int, num_freq_masks: int) -> torch.Tensor:
    _, f_dim, t = features.shape
    for _ in range(num_time_masks):
        w = int(torch.randint(1, time_mask + 1, (1,)))
        start = int(torch.randint(0, max(1, t - w) + 1, (1,)))
        features[:, :, start:start + w] = 0.0
    for _ in range(num_freq_masks):
        w = int(torch.randint(1, freq_mask + 1, (1,)))
        start = int(torch.randint(0, max(1, f_dim - w) + 1, (1,)))
        features[:, start:start + w, :] = 0.0
    return features


def prepare_batch(batch: dict, device):
    features = batch["inputs"].to(device).permute(0, 2, 1)
    lengths = torch.tensor(
        [getattr(c, "num_frames", None) or int(round(c.duration * 100))
         for c in batch["supervisions"]["cut"]],
        dtype=torch.long, device=device,
    )
    return features, lengths, extract_labels(batch, device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    for batch in loader:
        features, lengths, labels = prepare_batch(batch, device)
        functional.reset_net(model)
        logits = model(features, lengths)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="offline/conf/*.yaml")
    args = parser.parse_args()

    conf_path = PROJECT_ROOT / args.conf
    if not conf_path.exists():      # 兼容在 offline/ 目录里启动的写法（conf 相对 offline/）
        alt = Path(__file__).resolve().parent / args.conf
        if alt.exists():
            conf_path = alt
    with open(conf_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # GPU 选择：必须在任何 CUDA 调用之前设置（读取 conf 的 system.cuda_visible_devices）
    system = cfg.get("system") or {}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(system.get("cuda_visible_devices", "0"))

    fix_random_seed(int(cfg.get("seed", 312)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = PROJECT_ROOT / cfg["exp_dir"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(
                            str(exp_dir / "log-train-.rebuild")),
                            logging.StreamHandler()])
    logging.info("重建版离线训练启动：%s", json.dumps(cfg, ensure_ascii=False))
    model = AlphaSNNForKWS(dict(cfg["model"])).to(device)
    logging.info("模型参数: %.3fM", sum(p.numel() for p in model.parameters()) / 1e6)
    num_workers = int(cfg.get("num_workers", 8))
    train_loader = make_loader(PROJECT_ROOT / cfg["train_manifest"],
                               float(cfg["max_duration"]), shuffle=True,
                               num_workers=num_workers, persistent=True)
    test_loader = make_loader(PROJECT_ROOT / cfg["test_manifest"],
                              float(cfg["max_duration"]), num_workers=num_workers)

    # v2 稳定化实验：优化器类型可配置（候选 AdamW，原版类型未知）
    optimizer_type = str(cfg.get("optimizer_type", "adamw")).lower()
    opt_kwargs = dict(lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), **opt_kwargs)
    else:
        optimizer = optim.Adam(model.parameters(), **opt_kwargs)
    grad_clip = float(cfg.get("gradient_clip", 5.0))
    branch_balance_weight = float(cfg.get("branch_balance_weight", 0.01))
    num_epochs = int(cfg["num_epochs"])
    warmup = int(cfg.get("warmup_epochs", 30))
    t_max = int(cfg.get("cosine_t_max", num_epochs - warmup))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1))
    ce_loss = SpikingCELoss(cfg)   # 读取 conf 的 label_smoothing（默认 0.0），数值等价于 CE
    ema = ModelEMA(model, float(cfg.get("ema_decay", 0.9995))) if cfg.get("ema_decay") else None

    spec_aug_enabled = bool(cfg.get("spec_augment", False))
    spec_time_mask = int(cfg.get("spec_time_mask", 20))
    spec_freq_mask = int(cfg.get("spec_freq_mask", 5))
    spec_num_time_masks = int(cfg.get("spec_num_time_masks", 2))
    spec_num_freq_masks = int(cfg.get("spec_num_freq_masks", 2))
    if spec_aug_enabled:
        logging.info("SpecAugment 已启用: 时域遮罩 x%d（≤%d 帧），频域遮罩 x%d（≤%d 频带），仅作用于训练",
                     spec_num_time_masks, spec_time_mask, spec_num_freq_masks, spec_freq_mask)

    # KD 教师（可选）：TdnnLstm ANN，checkpoint 键无前缀
    teacher = None
    if cfg.get("kd_teacher"):
        from alpha.asr.kws.offline.kd_teacher import TdnnLstm

        teacher = TdnnLstm(num_features=int(cfg["model"]["n_inputs"]),
                           num_classes=int(cfg["model"]["n_outputs"]),
                           subsampling_factor=3).to(device)
        tck = torch.load(PROJECT_ROOT / cfg["kd_teacher"], map_location="cpu",
                         weights_only=True)
        teacher.load_state_dict(tck["model"] if "model" in tck else tck)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        logging.info("KD 教师已加载: %s（TdnnLstm ANN，alpha=%s, T=%s）",
                     cfg["kd_teacher"], cfg.get("kd_alpha", 1.0), cfg.get("kd_temperature", 2.0))

    best_test = best_ema_test = 0.0
    for epoch in range(num_epochs):
        model.train()
        if epoch < warmup:  # warmup + cosine
            lr = float(cfg["lr"]) * (epoch + 1) / max(warmup, 1)
            for g in optimizer.param_groups:
                g["lr"] = lr
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        tot_loss = tot_n = tot_correct = 0
        total_batches = getattr(train_loader, "est_batches", 0)   # 估算值（动态采样器无 len）
        epoch_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch + 1}/{num_epochs}",
                         unit="batch",
                         total=total_batches or None,
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                         miniters=max(1, total_batches // 20) if total_batches else 20,
                         mininterval=5)
        for batch_idx, batch in enumerate(epoch_bar):
            features, lengths, labels = prepare_batch(batch, device)
            if spec_aug_enabled:
                features = spec_augment(features, spec_time_mask, spec_freq_mask,
                                        spec_num_time_masks, spec_num_freq_masks)
            functional.reset_net(model)
            logits = model(features, lengths)
            loss, _ = ce_loss(logits, features, labels)
            if teacher is not None:
                functional.reset_net(teacher)
                with torch.no_grad():
                    t_logits = teacher(features)  # TdnnLstm 内部自带下采样，无需 lengths
                T = float(cfg.get("kd_temperature", 2.0))
                alpha = float(cfg.get("kd_alpha", 1.0))
                kd = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / T, dim=1),
                    nn.functional.softmax(t_logits / T, dim=1),
                    reduction="batchmean") * T * T
                loss = (1 - alpha) * loss + alpha * kd
            if branch_balance_weight > 0:   # v2：tau 分支均衡（防分支坍缩）
                loss = loss + branch_balance_weight * model.model.branch_balance_loss()
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:               # v2：梯度裁剪（抑制 warmup 后震荡）
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            tot_loss += loss.item() * len(labels)
            tot_n += len(labels)
            tot_correct += (logits.argmax(1) == labels).sum().item()
        train_acc = tot_correct / max(tot_n, 1)
        logging.info("Epoch %d: lr=%.2e loss=%.4f train_acc=%.2f%%",
                     epoch, lr, tot_loss / max(tot_n, 1), train_acc * 100)

        acc = evaluate(model, test_loader, device)
        if acc > best_test:                      # ← 移到日志前，best 与 EMA 行为一致
            best_test = acc
            torch.save({"model": model.state_dict(), "best_test_acc": acc,
                     "config": args.conf, "epoch": epoch}, exp_dir / "best-acc.pt")
        msg = f"Epoch {epoch}: raw test acc={acc:.2%} (best {best_test:.2%})"
        if ema is not None:
            ema_acc = evaluate(ema.ema_model, test_loader, device)
            if ema_acc > best_ema_test:      # EMA 权重在测试集上创新高 → 存档
                best_ema_test = ema_acc
                torch.save({"model": ema.ema_model.state_dict(), "best_test_acc": ema_acc,
                            "is_ema": True, "config": args.conf, "epoch": epoch},
                           exp_dir / "best-ema.pt")
            msg += f" | EMA test acc={ema_acc:.2%} (best {best_ema_test:.2%})"
        logging.info(msg)
        if acc > best_test:
            best_test = acc
            torch.save({"model": model.state_dict(), "best_test_acc": acc,
                        "config": args.conf, "epoch": epoch}, exp_dir / "best-acc.pt")
        torch.save({"model": model.state_dict(), "best_test_acc": best_test,
                    "config": args.conf, "epoch": epoch}, exp_dir / "last.pt")

    logging.info("训练完成：best raw=%.2f%%, best EMA=%.2f%%",
                 best_test * 100, best_ema_test * 100)


if __name__ == "__main__":
    main()
