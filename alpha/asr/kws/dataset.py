"""Lhotse cuts 数据集与 DataLoader 构建（29 类中文车载指令 KWS）。

数据为预计算 FBank（lilcom .lca），manifest 清单位于
examples/carKWS_offline/data/（离线训练+干净测试）与
examples/carKWS_online/data/（在线目标域）。

口径说明（评测对批组成敏感，任何一项改动都会改变精度）：
  - 离线训练/评测：DynamicBucketingSampler(max_duration=1500, num_buckets=30)
  - ANN 教师训练：DynamicBucketingSampler(max_duration=200, num_buckets=30) + lhotse SpecAugment
  - 在线 AdaKWS：SimpleCutSampler 按时长升序 + 每批固定条数（与旧 utils/data.py 语义一致）
"""

from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader


def load_cuts(manifest_dir, cuts_file: str) -> CutSet:
    """懒加载 lhotse CutSet 清单。manifest_dir 为空或 None 时直接用 cuts_file。"""
    if manifest_dir:
        return load_manifest_lazy(Path(manifest_dir) / cuts_file)
    return load_manifest_lazy(cuts_file)


def make_ann_spec_augment(conf: dict) -> SpecAugment:
    """ANN 教师的 SpecAugment（参数与原版训练日志一致）。"""
    return SpecAugment(
        time_warp_factor=int(conf.get('time_warp_factor', 80)),
        num_frame_masks=int(conf.get('num_frame_masks', 10)),
        frames_mask_size=int(conf.get('frames_mask_size', 100)),
        num_feature_masks=int(conf.get('num_feature_masks', 2)),
        features_mask_size=int(conf.get('features_mask_size', 27)),
    )


def _make_dataset(on_the_fly: bool = False, spec_augment: dict | None = None):
    """构建 K2SpeechRecognitionDataset（无 cut 变换；SpecAugment 仅训练集传入）。"""
    input_strategy = (
        OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        if on_the_fly else PrecomputedFeatures()
    )
    input_transforms = [make_ann_spec_augment(spec_augment)] if spec_augment else []
    return K2SpeechRecognitionDataset(
        cut_transforms=[],
        input_transforms=input_transforms,
        input_strategy=input_strategy,
        return_cuts=True,
    )


def build_train_loader(manifest_dir, cuts_file: str, max_duration: float,
                       num_buckets: int = 30, num_workers: int = 8, seed: int = 0,
                       spec_augment: dict | None = None,
                       on_the_fly: bool = False) -> DataLoader:
    """训练 loader：DynamicBucketing + shuffle。"""
    cuts = load_cuts(manifest_dir, cuts_file)
    dataset = _make_dataset(on_the_fly, spec_augment)
    sampler = DynamicBucketingSampler(
        cuts, max_duration=max_duration, shuffle=True,
        num_buckets=num_buckets, drop_last=False, seed=seed,
    )
    return DataLoader(dataset, sampler=sampler, batch_size=None,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=True)


def build_eval_loader(manifest_dir, cuts_file: str, max_duration: float,
                      num_buckets: int = 30, num_workers: int = 8,
                      on_the_fly: bool = False) -> DataLoader:
    """评测 loader：DynamicBucketing、不打乱（口径与训练期评测一致）。"""
    cuts = load_cuts(manifest_dir, cuts_file)
    dataset = _make_dataset(on_the_fly, None)
    sampler = DynamicBucketingSampler(
        cuts, max_duration=max_duration, shuffle=False,
        num_buckets=num_buckets, drop_last=False,
    )
    return DataLoader(dataset, sampler=sampler, batch_size=None,
                      num_workers=num_workers, pin_memory=True)


def build_adakws_dataloader(cuts: CutSet, cfg_target, batch_size: int) -> DataLoader:
    """AdaKWS 目标域 loader：固定每批条数 + 按时长升序（统一评测口径）。

    与原 utils/data.py 同语义：SimpleCutSampler、顺序固定，保证初始/最终
    精度出自完全一致的批次划分。
    """
    if batch_size < 1:
        raise ValueError(f'AdaKWS batch_size must be positive, got {batch_size}')
    dataset = _make_dataset(on_the_fly=bool(getattr(cfg_target, 'on_the_fly_feats', False)))
    sampler = SimpleCutSampler(
        cuts.sort_by_duration(ascending=True),
        max_duration=None,
        max_cuts=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=cfg_target.test_num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
