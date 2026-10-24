"""训练通用工具：标签提取、精度评测、预训练权重加载。"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from icefall.utils import AttributeDict
from spikingjelly.activation_based import functional


# ═══════════════════════════════════════════════════════════════
#  标签提取 / 精度评测
# ═══════════════════════════════════════════════════════════════

def extract_labels(batch: dict, device) -> torch.Tensor:
    """从 batch 的 supervisions 中提取说话人 id 作为标签（数据管线约定）。"""
    labels = [int(cut.supervisions[0].speaker) for cut in batch["supervisions"]["cut"]]
    return torch.tensor(labels, dtype=torch.long).to(device)


def batch_lengths(batch: dict, device) -> torch.Tensor:
    """每条 cut 的有效 FBank 帧数（num_frames 优先，回退 duration×100）。

    统一供离线训练（system.py）、评测（compute_test_accuracy）与
    在线适应（adakws_trainer）使用，避免三处重复实现。
    """
    return torch.tensor(
        [getattr(cut, "num_frames", None) or int(round(cut.duration * 100))
         for cut in batch["supervisions"]["cut"]],
        dtype=torch.long, device=device,
    )


def compute_test_accuracy(
    model: nn.Module,
    test_dl: torch.utils.data.DataLoader,
    device,
    world_size: int = 1,
) -> float:
    """计算测试集精度。传 lengths 做掩码池化，消除 batch 内补零帧污染。"""
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch in test_dl:
        feature = batch["inputs"].to(device).permute(0, 2, 1)
        lengths = batch_lengths(batch, device)
        functional.reset_net(model)
        class_logits = model(feature, lengths)
        labels = extract_labels(batch, device)
        total_correct += (class_logits.argmax(1) == labels).sum().item()
        total_samples += feature.size(0)

    if world_size > 1:
        import torch.distributed as dist

        stats = torch.tensor([total_correct, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_correct, total_samples = stats.cpu().tolist()

    return total_correct / total_samples if total_samples > 0 else 0.0


def _subset_of(source: str) -> str:
    """按录制源路径划分子集：真实车载 / 加噪 simu / RIR 混响。"""
    if "rec/comm_domain" in source:
        return "real_car"
    if "addsimu" in source:
        return "simu_noise"
    return "rir_reverb"


def compute_subset_accuracy(
    model: nn.Module,
    test_dl: torch.utils.data.DataLoader,
    device,
) -> dict:
    """按目标域子集统计精度（与整体精度同口径：eval + reset_net + 掩码池化）。"""
    model.eval()
    stats: dict = {}
    with torch.no_grad():
        for batch in test_dl:
            feature = batch["inputs"].to(device).permute(0, 2, 1)
            lengths = batch_lengths(batch, device)
            functional.reset_net(model)
            class_logits = model(feature, lengths)
            labels = extract_labels(batch, device)
            correct = (class_logits.argmax(1) == labels).cpu()
            for i, cut in enumerate(batch["supervisions"]["cut"]):
                key = _subset_of(cut.recording.sources[0].source)
                acc, total = stats.setdefault(key, [0, 0])
                stats[key] = [acc + int(correct[i].item()), total + 1]
    return {key: {"acc": acc / total, "n": total} for key, (acc, total) in stats.items()}


# ═══════════════════════════════════════════════════════════════
#  预训练权重加载
# ═══════════════════════════════════════════════════════════════

def load_pretrained_model(params: AttributeDict, model: nn.Module) -> None:
    """加载离线预训练权重到 MultiTauLIFCommanderKWS.backbone。

    checkpoint 结构: {"model": AlphaSNNForKWS.state_dict(), ...}
    参数名映射:  model.xxx      → backbone.xxx
                 subsample.xxx  → subsample.xxx（同名）
    其余前缀（如历史 checkpoint 的 domain_classifier.*）一律丢弃。
    """
    filename = Path(params.paths.pretrained_model)
    logging.info(f"尝试加载预训练模型: {filename}")
    logging.info(f"文件是否存在: {filename.exists()}")

    if not filename.exists():
        logging.warning(f"预训练模型文件不存在: {filename}")
        logging.info("将使用随机初始化的模型开始训练")
        return

    try:
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        src_state = checkpoint["model"] if "model" in checkpoint else checkpoint

        mapped = {}
        for key, value in src_state.items():
            if key.startswith("model."):
                mapped["backbone." + key[6:]] = value
            elif key.startswith("backbone."):
                # AdaKWS 产物的 checkpoint 直接以 backbone.* 保存，直通加载
                mapped[key] = value
            elif key.startswith("subsample."):
                mapped[key] = value

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logging.warning(f"缺失的参数（将被随机初始化）: {missing[:5]}...")
        logging.info(f"成功加载 {len(mapped)} 个预训练参数（共 {len(src_state)} 个）")

    except Exception as e:
        logging.warning(f"加载预训练模型失败: {e}")
        logging.info("将使用随机初始化的模型开始训练")

    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
