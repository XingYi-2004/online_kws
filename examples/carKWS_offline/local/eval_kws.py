"""固定口径评测（29 类中文车载指令，干净测试 357 条）。

评测口径（任何一项改动精度都会变，勿改）：
  lhotse DynamicBucketingSampler, max_duration=1500s, num_buckets=30,
  drop_last=False, shuffle=False, seed=312（与 experiment4 训练期 test loader 一致）。

用法（cd AlphaSNN）：
  python examples/carKWS_offline/local/eval_kws.py --ckpt <best-acc.ckpt> \
      [--ckpt-ema <best-ema.ckpt>] [--out <eval.log路径>]

终端打印的全部内容会原样镜像到 <ckpt>/../eval.log
（即与被评测实验的 log.log / hparams.yaml 同目录），每次评测重新生成。

金标准（迁移验证用，历史 {"model": ...} 格式直接可评）：
  --ckpt alphasnn_29class_kd_rebuild_v2/experiment4/best-acc.pt      → 350/357 = 98.04%
  --ckpt-ema alphasnn_29class_kd_rebuild_v2/experiment4/best-ema.pt  → 350/357 = 98.04%
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


class Tee:
    """终端输出镜像：write 到原流的同时写入共享的镜像文件。"""

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, msg):
        self.stream.write(msg)
        self.file.write(msg)
        return len(msg)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def run_eval(args):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='raw 权重（Lightning .ckpt 或 {"model": ...} 格式）')
    parser.add_argument('--ckpt-ema', default=None, help='EMA 权重（可选）')
    parser.add_argument('--out', default=None,
                        help='终端输出镜像的落盘路径'
                             '（默认 <ckpt>/../eval.log，与被评测实验的'
                             ' log.log / hparams.yaml 同目录）')
    args = parser.parse_args()

    log_path = Path(args.out) if args.out else \
        Path(args.ckpt).resolve().parent.parent / 'eval.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[eval.log] 终端输出镜像: {log_path}')

    real_stdout, real_stderr = sys.stdout, sys.stderr
    with open(log_path, 'w', encoding='utf-8') as log_file:   # 单一句柄，两个 Tee 共享
        sys.stdout = Tee(real_stdout, log_file)
        sys.stderr = Tee(real_stderr, log_file)
        try:
            run_eval(args)
        finally:
            log_file.flush()
            sys.stdout, sys.stderr = real_stdout, real_stderr


if __name__ == '__main__':
    main()
