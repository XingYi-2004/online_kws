"""AdaKWS-style source-free test-time adaptation for the MT-SNN model.

The target labels are never read by this trainer.  It adapts only the affine
parameters of BatchNorm layers using selective entropy minimization and the
prediction-probability drop after feature masking (PKC/PLPD).

历史说明: 教师模型/EMA、目标域记忆库、源域回放 CE、输入谱增益、BCT 先验校正
等叠加组件均已随实验清理（均未超过纯 AdaKWS），记录见 AlphaSNN/README.md。
"""

import copy
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from alpha.asr.kws.utils.train_utils import batch_lengths as _batch_lengths


def _set_adaptation_parameters(
    model: nn.Module, bn_mode: str, update_scope: str
) -> list[nn.Parameter]:
    """Freeze MT-SNN and enable the configured AdaKWS update scope."""
    valid_bn_modes = {"no_running", "train_running", "checkpoint_running"}
    if bn_mode not in valid_bn_modes:
        raise ValueError(
            f"不支持的 adakws_bn_mode={bn_mode!r}，可选值为 {sorted(valid_bn_modes)}"
        )
    valid_scopes = {"bn_affine", "bn_gn_affine", "bn_classifier", "last_block_classifier"}
    if update_scope not in valid_scopes:
        raise ValueError(
            f"不支持的 adakws_update_scope={update_scope!r}，可选值为 "
            f"{sorted(valid_scopes)}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    core = model.module if hasattr(model, "module") else model

    if bn_mode == "no_running":
        for module in core.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                # Match the official AdaKWS/TENT configuration: use each
                # batch's statistics and do not consume source statistics.
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    if update_scope in {"bn_affine", "bn_gn_affine", "bn_classifier"}:
        norm_types = (nn.modules.batchnorm._BatchNorm,)
        if update_scope == "bn_gn_affine":
            norm_types = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)
        for module in core.modules():
            if isinstance(module, norm_types):
                if module.weight is not None:
                    module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True

    if update_scope in {"bn_classifier", "last_block_classifier"}:
        for parameter in core.backbone.classifier.parameters():
            parameter.requires_grad = True

    if update_scope == "last_block_classifier":
        for parameter in core.backbone.blocks[-1].parameters():
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return trainable


def _set_adaptation_mode(model: nn.Module, bn_mode: str) -> None:
    """Keep SNN training mode while explicitly choosing the BN statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            # The SNN must stay in train mode for surrogate gradients, but BN
            # can independently use frozen checkpoint statistics for Exp. B.
            module.train(mode=bn_mode != "checkpoint_running")


def _mask_features(
    features: torch.Tensor,
    time_mask: int,
    freq_mask: int,
    num_time_masks: int,
    num_freq_masks: int,
) -> torch.Tensor:
    """Create the PKC view with time/frequency masking."""
    transformed = features.clone()
    _, num_features, num_frames = transformed.shape

    for _ in range(num_time_masks):
        width = random.randint(1, min(time_mask, num_frames))
        start = random.randint(0, num_frames - width)
        transformed[:, :, start : start + width] = 0.0
    for _ in range(num_freq_masks):
        width = random.randint(1, min(freq_mask, num_features))
        start = random.randint(0, num_features - width)
        transformed[:, start : start + width, :] = 0.0
    return transformed


def _entropy(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature != 1.0:
        probabilities = F.softmax(torch.log(probabilities.clamp_min(1e-8)) / temperature, dim=1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)


def _adakws_loss(
    model: nn.Module,
    features: torch.Tensor,
    lengths: torch.Tensor,
    cfg,
):
    """Return AdaKWS loss, selection counts, and mean batch entropy.

    两级筛选（论文 Eq.5 的选择机制）：
      1) 熵筛选：只保留模型"笃定"的样本（熵 < 阈值）；
      2) PKC/PLPD 筛选：遮罩时频区域后原类概率下降 > 阈值的样本才保留
         （说明模型判断确实依赖语音内容，而非碰运气）。
    损失 = 加权熵最小化：权重 = exp(sigma - H) + exp(drop)，熵越小、
    遮罩掉置信度越多的样本话语权越大。
    """
    functional.reset_net(model)
    logits = model(features, lengths)
    probabilities = F.softmax(logits, dim=1)
    entropy = _entropy(probabilities, float(cfg.adakws_temperature))
    pseudo_labels = probabilities.argmax(dim=1)
    original_confidence = probabilities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)

    entropy_selected = entropy < float(cfg.adakws_entropy_threshold)
    if not entropy_selected.any():
        return None, {"entropy": 0, "pkc": 0, "final": 0}, entropy.detach().mean().item()

    entropy_features = features[entropy_selected]
    entropy_lengths = lengths[entropy_selected]
    entropy_pseudo_labels = pseudo_labels[entropy_selected]
    entropy_confidence = original_confidence[entropy_selected]

    masked_features = _mask_features(
        entropy_features,
        int(cfg.adakws_time_mask),
        int(cfg.adakws_freq_mask),
        int(cfg.adakws_num_time_masks),
        int(cfg.adakws_num_freq_masks),
    )
    with torch.no_grad():
        functional.reset_net(model)
        masked_logits = model(masked_features, entropy_lengths)
        masked_probabilities = F.softmax(masked_logits, dim=1)
        masked_confidence = masked_probabilities.gather(
            1, entropy_pseudo_labels.unsqueeze(1)
        ).squeeze(1)

    probability_drop = entropy_confidence - masked_confidence
    pkc_selected = probability_drop > float(cfg.adakws_pkc_threshold)
    selected_entropy = entropy[entropy_selected][pkc_selected]
    counts = {
        "entropy": int(entropy_selected.sum().item()),
        "pkc": int(pkc_selected.sum().item()),
        "final": int(pkc_selected.sum().item()),
    }
    if not pkc_selected.any():
        return None, counts, entropy.detach().mean().item()

    # Eq. (5) in AdaKWS: entropy and PKC/PLPD both contribute to weighting.
    drop_selected = probability_drop[pkc_selected]
    sigma = float(cfg.adakws_sigma)
    weights = torch.exp(
        torch.clamp(sigma - selected_entropy, min=-20.0, max=20.0)
    ) + torch.exp(torch.clamp(drop_selected, min=-20.0, max=20.0))
    loss = (weights * selected_entropy).mean()
    return loss, counts, entropy.detach().mean().item()


def train_adakws(
    params,
    model: nn.Module,
    target_dl,
    test_dl,
    device,
    tb_writer: Optional[SummaryWriter] = None,
):
    """Run AdaKWS adaptation using only unlabeled target inputs.

    只更新 adakws_update_scope 圈定的参数（默认 6 个 BN 层的 affine，4608 个），
    其余全部冻结；按无监督熵评分保留最佳迭代并保存 best-adakws.pt。
    """
    cfg = params.online
    # Backward-compatible mapping for experiments A/B recorded before
    # the explicit three-way BN mode was added.
    configured_bn_mode = getattr(cfg, "adakws_bn_mode", None)
    if configured_bn_mode is None:
        bn_mode = "no_running" if bool(getattr(cfg, "adakws_use_batch_stats", True)) else "checkpoint_running"
    else:
        bn_mode = str(configured_bn_mode).lower()
    update_scope = str(getattr(cfg, "adakws_update_scope", "bn_affine")).lower()

    trainable = _set_adaptation_parameters(model, bn_mode, update_scope)
    if not trainable:
        raise RuntimeError("AdaKWS requires at least one BatchNorm affine parameter")

    optimizer_name = str(getattr(cfg, "adakws_optimizer", "sgd")).lower()
    optimizer_kwargs = {
        "lr": float(cfg.adakws_lr),
        "weight_decay": float(cfg.adakws_weight_decay),
    }
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable,
            momentum=float(getattr(cfg, "adakws_momentum", 0.9)),
            nesterov=bool(getattr(cfg, "adakws_nesterov", True)),
            **optimizer_kwargs,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable, **optimizer_kwargs)
    else:
        raise ValueError(
            f"不支持的 adakws_optimizer={optimizer_name!r}，可选值为 'sgd' 或 'adam'"
        )
    logging.info(
        "开始 AdaKWS 在线适应：更新范围=%s，共 %d 个参数；"
        "目标域标签不参与适应；优化器=%s；BN统计模式=%s",
        update_scope,
        sum(parameter.numel() for parameter in trainable),
        optimizer_name,
        bn_mode,
    )

    best_unsupervised_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_iteration = -1
    num_iterations = int(cfg.num_iterations)

    for iteration in range(num_iterations):
        _set_adaptation_mode(model, bn_mode)
        total_loss = 0.0
        total_entropy_selected = 0
        total_pkc_selected = 0
        total_selected = 0
        total_batches = 0

        for batch in target_dl:
            features = batch["inputs"].to(device).permute(0, 2, 1)
            lengths = _batch_lengths(batch, device)
            for _ in range(int(cfg.adakws_steps)):
                _set_adaptation_mode(model, bn_mode)
                loss, counts, mean_entropy = _adakws_loss(model, features, lengths, cfg)
                total_entropy_selected += counts["entropy"]
                total_pkc_selected += counts["pkc"]
                total_selected += counts["final"]
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.grad_clip))
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                params.batch_idx_train += 1

                if tb_writer:
                    tb_writer.add_scalar("adakws/loss", loss.item(), params.batch_idx_train)
                    tb_writer.add_scalar("adakws/entropy_selected", counts["entropy"], params.batch_idx_train)
                    tb_writer.add_scalar("adakws/pkc_selected", counts["pkc"], params.batch_idx_train)
                    tb_writer.add_scalar("adakws/entropy", mean_entropy, params.batch_idx_train)

        unsupervised_score = total_loss / max(total_batches, 1)
        logging.info(
            "AdaKWS 迭代 %d/%d：熵筛选 %d 条，PKC筛选 %d 条，"
            "最终共同筛选 %d 条，平均熵损失 %.6f，无监督评分 %.6f",
            iteration,
            num_iterations - 1,
            total_entropy_selected,
            total_pkc_selected,
            total_selected,
            total_loss / max(total_batches, 1),
            unsupervised_score,
        )

        # 目标域标签从不参与该评分，按无监督评分保留最佳迭代
        if total_batches > 0 and unsupervised_score < best_unsupervised_score:
            best_unsupervised_score = unsupervised_score
            best_state = copy.deepcopy(model.state_dict())
            best_iteration = iteration

    model.load_state_dict(best_state)
    logging.info("AdaKWS 适应完成：按无监督评分保留迭代 %d 的模型", best_iteration)

    best_path = Path(params.system.exp_dir) / "best-adakws.pt"
    torch.save({"model": model.state_dict()}, best_path)
    logging.info("AdaKWS 最佳模型已保存: %s", best_path)
