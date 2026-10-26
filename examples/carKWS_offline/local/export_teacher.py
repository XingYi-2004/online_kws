"""ANN 教师 epoch 平均导出 + 评测（对应原版 decode_ann --epoch 17 --avg 9 --export True）。

对 save_top_k: -1 全量保存的 Lightning 逐 epoch checkpoint 做权重平均。
输出默认放回被平均的 checkpoints 目录（与原版 pretrained.pt 存于训练目录一致）：
  --ckpt-dir exp/logs/carKWS_ANN/ann01/checkpoints
  → exp/logs/carKWS_ANN/ann01/checkpoints/pretrained.pt（--out 可覆盖）
导出后自动在干净测试集上评测精度。

评测口径（TdnnLstm 无掩码全帧均值池化，精度对批组成/padding 敏感，勿改）：
  lhotse DynamicBucketingSampler, max_duration=200s, num_buckets=30,
  shuffle=False, seed=42，数据 = examples/carKWS_offline/data 干净测试 357 条。

用法（cd AlphaSNN）：
  python examples/carKWS_offline/local/export_teacher.py \
      --ckpt-dir exp/logs/carKWS_ANN/ann01/checkpoints \
      --start-epoch 9 --end-epoch 17
  # → exp/logs/carKWS_ANN/ann01/checkpoints/pretrained.pt
  python examples/carKWS_offline/local/export_teacher.py --eval-only \
      --out exp/logs/carKWS_ANN/ann01/checkpoints/pretrained.pt   # 复测已有平均权重

期望：平均教师 Test acc ≈99.7%。
（学生 SNN 权重无需导出：在线/评测加载器直接读 Lightning best-acc.ckpt /
 best-ema.ckpt，见 kws/utils/train_utils.load_pretrained_model）
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np

if not hasattr(np, 'int'):
    np.int = int

import torch
from lhotse.utils import fix_random_seed

from alpha.asr.kws.dataset import build_eval_loader
from alpha.asr.kws.utils.train_utils import extract_labels

MANIFEST_DIR = PROJECT_ROOT / 'examples/carKWS_offline/data'
CUTS_FILE = 'command_cuts_test_29.jsonl'
MAX_DURATION = 200      # 与 ANN 训练期口径一致
NUM_BUCKETS = 30
SEED = 42


def strip_prefix(state_dict, prefix='model.'):
    return {k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def average_epochs(ckpt_dir: Path, start: int, end: int) -> dict:
    avg = None
    epochs = list(range(start, end + 1))
    for e in epochs:
        path = ckpt_dir / f'epoch={e}.ckpt'
        if not path.exists():
            raise FileNotFoundError(f'缺少 {path}（需 save_top_k: -1 全量保存）')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = strip_prefix(ckpt['state_dict'])
        if avg is None:
            avg = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    for k in avg:
        avg[k] /= len(epochs)
    return avg


@torch.no_grad()
def evaluate_teacher(model, loader, device):
    """教师是普通 ANN（无 SNN 状态），全帧均值池化，前向不带 lengths。"""
    model.eval()
    correct = total = 0
    for batch in loader:
        feature = batch['inputs'].to(device).permute(0, 2, 1).float()  # [B,F,T]
        logits = model(feature)
        labels = extract_labels(batch, device)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', default=None, help='checkpoints 目录')
    parser.add_argument('--start-epoch', type=int, default=9)
    parser.add_argument('--end-epoch', type=int, default=17)
    parser.add_argument('--out', default=None,
                        help='输出路径（默认 <ckpt-dir>/pretrained.pt，'
                             '与被平均权重同目录）')
    parser.add_argument('--eval-only', action='store_true',
                        help='跳过平均，直接评测 --out 指向的已有平均权重')
    args = parser.parse_args()

    if args.eval_only:
        out_path = Path(args.out) if args.out else None
        if out_path is None:
            parser.error('--eval-only 需要 --out 指向已有平均权重')
        ckpt = torch.load(out_path, map_location='cpu', weights_only=True)
        avg = ckpt['model'] if 'model' in ckpt else ckpt
    else:
        if not args.ckpt_dir:
            parser.error('平均模式需要 --ckpt-dir（或用 --eval-only 评测已有权重）')
        ckpt_dir = Path(args.ckpt_dir)
        out_path = Path(args.out) if args.out else ckpt_dir / 'pretrained.pt'
        avg = average_epochs(ckpt_dir, args.start_epoch, args.end_epoch)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': avg}, out_path)
        print(f'[export] teacher avg epoch {args.start_epoch}-{args.end_epoch}: '
              f'-> {out_path}')

    # ---- 导出后必评：平均教师的精度决定 300 epoch KD 是否值得开跑 ----
    from alpha.asr.model.tdnn_lstm import TdnnLstm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TdnnLstm(num_features=80, num_classes=29, subsampling_factor=3)
    model.load_state_dict(avg)
    model.to(device)

    fix_random_seed(SEED)
    cuts = build_eval_loader(MANIFEST_DIR, CUTS_FILE, MAX_DURATION,
                             num_buckets=NUM_BUCKETS, num_workers=4)
    correct, total = evaluate_teacher(model, cuts, device)
    print(f'[teacher] Test acc = {correct / total:.2%} ({correct}/{total} cuts)')


if __name__ == '__main__':
    main()
