"""数据加载：Lhotse cuts 加载、数据集与 DataLoader 构建。"""

import logging
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader


def load_cuts(manifest_dir, cuts_file: str) -> CutSet:
    return load_manifest_lazy(Path(manifest_dir) / cuts_file)


def _make_dataset(cfg_target, on_the_fly: bool) -> K2SpeechRecognitionDataset:
    """构建 K2SpeechRecognitionDataset（无 cut 变换；SpecAugment 不用于在线适应）。

    注意：offline/train_multitau.py 与 offline/eval_multitau.py 跨模块复用本函数
    （_make_dataset(None, False)），修改签名前先确认离线侧调用。
    """
    input_strategy = (
        OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))) if on_the_fly else PrecomputedFeatures()
    )
    return K2SpeechRecognitionDataset(
        cut_transforms=[],
        input_transforms=[],
        input_strategy=input_strategy,
        return_cuts=True,
    )


def build_test_dataloader(cuts: CutSet, cfg_target) -> DataLoader:
    """训练时口径的测试 dataloader（DynamicBucketing，不打乱；offline/eval_multitau.py 使用）。"""
    dataset = _make_dataset(cfg_target, cfg_target.on_the_fly_feats)
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=cfg_target.max_duration,
        shuffle=False,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=cfg_target.test_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def build_adakws_dataloader(cuts: CutSet, cfg_target, batch_size: int) -> DataLoader:
    """AdaKWS target loader with a fixed number of cuts per adaptation batch.

    也是统一评测口径的来源：固定每批条数 + 按时长升序排序（相邻样本时长接近，
    减少补零），顺序固定，保证初始/验证/最终精度出自完全一致的批次划分。
    """
    if batch_size < 1:
        raise ValueError(f"AdaKWS batch_size must be positive, got {batch_size}")

    dataset = _make_dataset(cfg_target, cfg_target.on_the_fly_feats)
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
