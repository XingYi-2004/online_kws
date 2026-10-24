"""固定口径评测（29 类中文车载指令，干净测试 357 条）。

评测口径（任何一项改动精度都会变，勿改）：
  lhotse DynamicBucketingSampler, max_duration=1500s, num_buckets=30,
  drop_last=False, shuffle=False, seed=312（与 experiment4 训练期 test loader 一致）。

用法（cd AlphaSNN）：
  python examples/carKWS_offline/local/eval_kws.py --ckpt <best-acc.pt> \
      [--ckpt-ema <best-ema.pt>]

金标准（迁移验证用）：
  --ckpt alphasnn_29class_kd_rebuild_v2/experiment4/best-acc.pt      → 350/357 = 98.04%
  --ckpt-ema alphasnn_29class_kd_rebuild_v2/experiment4/best-ema.pt  → 350/357 = 98.04%
（输入为 export_kws.py 导出的 {"model": ...} 格式，历史 checkpoint 同格式可直接评）
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
    np.int = int  # numpy 2.x 兼容 spikingjelly cupy 后端

import torch
from lhotse.utils import fix_random_seed

from alpha.asr.kws.dataset import build_eval_loader, load_cuts
from alpha.asr.model.multitau_lif_commander_kws import MultiTauLIFCommanderKWS
from alpha.asr.kws.utils.config import load_config
from alpha.asr.kws.utils.train_utils import compute_test_accuracy, load_pretrained_model

ONLINE_CONF = PROJECT_ROOT / 'examples/carKWS_online/conf/online_adakws.yaml'
MANIFEST_DIR = PROJECT_ROOT / 'examples/carKWS_offline/data'
CUTS_FILE = 'command_cuts_test_29.jsonl'
MAX_DURATION = 1500
NUM_BUCKETS = 30
SEED = 312


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='raw 权重（Lightning .ckpt 或 {"model": ...} 格式）')
    parser.add_argument('--ckpt-ema', default=None, help='EMA 权重（可选）')
    args = parser.parse_args()

    params = load_config(ONLINE_CONF)  # 仅取 model 块与加载工具
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fix_random_seed(SEED)

    cuts = load_cuts(MANIFEST_DIR, CUTS_FILE)
    print(f'clean test set: {len(cuts)} cuts | seed={SEED} '
          f'max_duration={MAX_DURATION} num_buckets={NUM_BUCKETS}')
    loader = build_eval_loader(MANIFEST_DIR, CUTS_FILE, MAX_DURATION,
                               num_buckets=NUM_BUCKETS, num_workers=4)

    for tag, ckpt in [('raw', args.ckpt), ('EMA', args.ckpt_ema)]:
        if not ckpt:
            continue
        # .ckpt 为 Lightning 格式：EMA 项需物化 ema_state_dict（use_ema）
        params.paths.use_ema = (tag == 'EMA') and ckpt.endswith('.ckpt')
        model = MultiTauLIFCommanderKWS(dict(params.model)).to(device)
        params.paths.pretrained_model = str(Path(ckpt).resolve())
        load_pretrained_model(params, model)
        acc = compute_test_accuracy(model, loader, device)
        print(f'[{tag}] acc = {acc:.2%}  ({ckpt})')


if __name__ == '__main__':
    main()
