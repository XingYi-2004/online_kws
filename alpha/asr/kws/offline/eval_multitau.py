"""离线评测脚本：MultiTauLIFCommanderKWS（29 类中文车载指令，干净测试集 357 条）。

一条命令复现重建版 98.04%（checkpoint 与评测口径全部固定在脚本内，无参数）：

  cd AlphaSNN && conda run -n snn python alpha/asr/kws/offline/eval_multitau.py

输出（与训练日志逐位一致）：
  [raw] 350/357 = 98.04%（checkpoint 记录 98.04%，E224）
  [EMA] 350/357 = 98.04%（checkpoint 记录 98.04%，E253）

评测口径与训练脚本 test loader 完全一致，改动任何一项精度都会变：
  lhotse DynamicBucketingSampler, max_duration=1500s, num_buckets=30,
  drop_last=False, seed=312。
  ⚠️ num_buckets 的 lhotse 默认是 10！与训练脚本的 30 不一致会导致同一
  checkpoint 精度差 1~2pp——这正是"98.04 复测差 2~3pp"悬案的根因（实测教训）。
"""

import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "int"):
    np.int = int

# AlphaSNN 框架根（offline → kws → asr → alpha → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from lhotse.dataset import DynamicBucketingSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from alpha.asr.kws.model_kws import MultiTauLIFCommanderKWS
from alpha.asr.kws.utils.config import load_config
from alpha.asr.kws.utils.data import _make_dataset, load_cuts
from alpha.asr.kws.utils.train_utils import extract_labels, load_pretrained_model

# 在线配置仅取 model/data.target 块（评测封装与口径同在线管线）
KWS_CONF = Path(__file__).resolve().parents[1] / "conf" / "online_adakws.yaml"

CHECKPOINT = "alphasnn_29class_kd_rebuild_v2/experiment4/best-acc.pt"      # raw 98.04%（E224）
EMA_CHECKPOINT = "alphasnn_29class_kd_rebuild_v2/experiment4/best-ema.pt"  # EMA 98.04%（E253）
MANIFEST = "data/fbank/command_cuts_test_29.jsonl"  # 干净测试集 357 条（29 类）
SEED = 312  # 训练脚本同款


@torch.no_grad()
def evaluate(model, loader, device):
    """整集精度：eval 模式 + 每批 reset_net + lengths 掩码前向。"""
    model.eval()
    total = correct = 0
    for batch in loader:
        features = batch["inputs"].to(device).permute(0, 2, 1)
        lengths = torch.tensor(
            [getattr(c, "num_frames", None) or int(round(c.duration * 100))
             for c in batch["supervisions"]["cut"]],
            dtype=torch.long, device=device,
        )
        functional.reset_net(model)
        logits = model(features, lengths)
        labels = extract_labels(batch, device)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return correct, total


def load_model(checkpoint, params, device):
    """构建模型并加载 checkpoint 权重（键映射 model.* → backbone.*）。"""
    model = MultiTauLIFCommanderKWS(dict(params.model)).to(device)
    params.paths.pretrained_model = checkpoint
    load_pretrained_model(params, model)
    return model





def main():
    # ⚠️ 评测口径三要素，与训练脚本 test loader 逐项对齐，勿改：
    #   max_duration=1500 / num_buckets=30（lhotse 默认 10！）/ seed=312
    fix_random_seed(SEED)
    params = load_config(KWS_CONF)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = PROJECT_ROOT / MANIFEST
    cuts = load_cuts(manifest_path.parent, manifest_path.name)
    dataset = _make_dataset(params.data.target, False)
    sampler = DynamicBucketingSampler(cuts, max_duration=1500.0, shuffle=False,
                                      num_buckets=30, drop_last=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None,
                        num_workers=2, pin_memory=True)

    correct, total = evaluate(load_model(CHECKPOINT, params, device), loader, device)
    print(f"[raw] {correct}/{total} = {correct / total:.2%}")

    correct, total = evaluate(load_model(EMA_CHECKPOINT, params, device), loader, device)
    print(f"[EMA] {correct}/{total} = {correct / total:.2%}")


if __name__ == "__main__":
    main()
