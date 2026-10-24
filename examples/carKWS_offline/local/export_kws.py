"""Lightning checkpoint → 在线管线兼容格式导出。

学生（SNN KD）：
  python examples/carKWS_offline/local/export_kws.py \
      --ckpt exp/logs/carKWS_SNN_KD/kd01/checkpoints/best-acc.ckpt \
      --ckpt-ema exp/logs/carKWS_SNN_KD/kd01/checkpoints/best-ema.ckpt \
      --out exp/logs/carKWS_SNN_KD/kd01/checkpoints/export
  产出 best-acc.pt / best-ema.pt，格式 {"model": state_dict, "epoch": ...,
  "best_test_acc": ...}，键名 model.*，在线 load_pretrained_model 直接加载
  （model.* → backbone.* 映射）。

教师（ANN，epoch 平均）：
  python examples/carKWS_offline/local/export_kws.py --mode teacher \
      --ckpt-dir exp/logs/carKWS_ANN/ann01/checkpoints \
      --start-epoch 9 --end-epoch 17 --out kd_teacher/pretrained.pt
  对应原版 decode_ann --epoch 17 --avg 9（epoch 9–17 checkpoint 平均）。
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
from omegaconf import OmegaConf


def strip_prefix(state_dict, prefix='model.'):
    return {k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def export_student(ckpt_path: Path, out_dir: Path, tag: str, is_ema: bool):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if not is_ema:
        # raw 权重：Lightning state_dict 的 model.* 键即历史格式，直接过滤导出
        sd = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    else:
        # EMA 权重：需重建同构模型，把 EMA 换入后导出
        from alpha.asr.kws.system import MultiTauKwsKd
        from torch_ema import ExponentialMovingAverage

        conf = OmegaConf.create(ckpt.get('hyper_parameters') or {})
        if 'model' not in conf:
            raise RuntimeError(
                f'{ckpt_path} 缺少 hyper_parameters，请用 --conf 指定训练配置')
        system = MultiTauKwsKd(conf)
        system.load_state_dict(ckpt['state_dict'])
        if 'ema_state_dict' not in ckpt:
            raise RuntimeError(f'{ckpt_path} 无 ema_state_dict（该次训练未启用 EMA）')
        ema = ExponentialMovingAverage(system.parameters(), decay=0.9995)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.copy_to(system.parameters())
        sd = {f'model.{k}': v for k, v in system.model.state_dict().items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{tag}.pt'
    torch.save({'model': sd, 'epoch': int(ckpt.get('epoch', -1)), 'is_ema': is_ema},
               out_path)
    print(f'[export] {tag}: {ckpt_path} -> {out_path} '
          f'({len(sd)} tensors, epoch={ckpt.get("epoch")})')


def export_teacher(ckpt_dir: Path, start: int, end: int, out_path: Path):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': avg}, out_path)
    print(f'[export] teacher avg epoch {epochs}: -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='student', choices=['student', 'teacher'])
    parser.add_argument('--ckpt', default=None, help='student: raw Lightning ckpt')
    parser.add_argument('--ckpt-ema', default=None, help='student: EMA Lightning ckpt')
    parser.add_argument('--ckpt-dir', default=None, help='teacher: checkpoints 目录')
    parser.add_argument('--start-epoch', type=int, default=9)
    parser.add_argument('--end-epoch', type=int, default=17)
    parser.add_argument('--out', default=None, help='输出文件/目录')
    args = parser.parse_args()

    if args.mode == 'teacher':
        out = Path(args.out or 'kd_teacher/pretrained.pt')
        export_teacher(Path(args.ckpt_dir), args.start_epoch, args.end_epoch, out)
        return

    if not args.ckpt and not args.ckpt_ema:
        parser.error('student 模式需要 --ckpt 和/或 --ckpt-ema')
    if args.ckpt:
        out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / 'export'
        export_student(Path(args.ckpt), out_dir, 'best-acc', is_ema=False)
    if args.ckpt_ema:
        out_dir = Path(args.out) if args.out else Path(args.ckpt_ema).parent / 'export'
        export_student(Path(args.ckpt_ema), out_dir, 'best-ema', is_ema=True)


if __name__ == '__main__':
    main()
