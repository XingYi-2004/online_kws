"""Multi-tau LIF memory backbone for keyword spotting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from omegaconf import DictConfig, OmegaConf

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode

from modules.utils.logging import logger
from modules.blocks.spike_fsmn.multi_tau_lif import MultiTauLIFMemory


class BatchMajorLIF(nn.Module):
    """Apply a multi-step LIF node to batch-major sequences."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.node = LIFNode(
            tau=float(config.get("init_tau", 2.0)),
            v_threshold=float(config.get("v_threshold", 1.0)),
            v_reset=_lif_v_reset(config),
            surrogate_function=_resolve_surrogate(config),
            detach_reset=bool(config.get("detach_reset", False)),
            step_mode="m",
            decay_input=bool(config.get("decay_input", True)),
            store_v_seq=False,
            backend=config.get("backend", "torch"),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x to be 3-D (B, T, D), got {tuple(x.shape)}")
        return self.node(x.transpose(0, 1)).transpose(0, 1).contiguous()


class SequenceNorm(nn.Module):
    """Normalize batch-major sequence features with BN or LN."""

    def __init__(self, dim: int, norm_type: str = "bn") -> None:
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_type in {"none", "identity"}:
            self.norm = nn.Identity()
        else:
            raise ValueError("norm_type must be one of: bn, ln, none")

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_type == "bn":
            return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.norm(x)


class SEELiteFrontend(nn.Module):
    """Lightweight acoustic frontend: pointwise conv, temporal depthwise conv, norm, LIF."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        in_dim = int(config["n_inputs"])
        dim = int(config["n_hidden_neurons"])
        kernel_size = int(config.get("frontend_kernel_size", 7))
        dilation = int(config.get("frontend_dilation", 1))
        if kernel_size % 2 == 0:
            raise ValueError("frontend_kernel_size must be odd")
        if dilation < 1:
            raise ValueError("frontend_dilation must be >= 1")
        self.in_dim = in_dim
        self.pointwise = nn.Conv1d(
            in_dim,
            dim,
            kernel_size=1,
            bias=bool(config.get("frontend_pw_bias", True)),
        )
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            groups=dim,
            bias=bool(config.get("frontend_dw_bias", False)),
        )
        self.norm = SequenceNorm(dim, _norm_type(config))
        self.lif = BatchMajorLIF(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_input(x, self.in_dim)
        y = x.transpose(1, 2)
        y = self.pointwise(y)
        y = self.depthwise(y).transpose(1, 2).contiguous()
        y = self.norm(y)
        y = self.lif(y)
        return _apply_attention_mask(y, attention_mask)


class LinearFrontend(nn.Module):
    """Frame-wise linear projection frontend for light ablations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.in_dim = int(config["n_inputs"])
        dim = int(config["n_hidden_neurons"])
        self.linear = nn.Linear(self.in_dim, dim, bias=bool(config.get("frontend_linear_bias", True)))
        self.norm = SequenceNorm(dim, _norm_type(config))
        self.lif = BatchMajorLIF(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_input(x, self.in_dim)
        y = self.linear(x)
        y = self.norm(y)
        y = self.lif(y)
        return _apply_attention_mask(y, attention_mask)


class ChannelMixer(nn.Module):
    """Lightweight per-frame channel fusion for sparse memory features."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = int(config["n_hidden_neurons"])
        self.norm = SequenceNorm(dim, _norm_type(config))
        self.linear = nn.Linear(dim, dim, bias=bool(config.get("channel_mixer_bias", True)))
        self.lif = BatchMajorLIF(config)

    def forward(self, x: Tensor) -> Tensor:
        return self.lif(self.linear(self.norm(x)))


class GroupedChannelMixer(nn.Module):
    """Grouped pointwise channel mixer with channel shuffle."""

    def __init__(self, dim: int, groups: int = 4, bias: bool = False) -> None:
        super().__init__()
        self.groups = _validate_group_count(dim, groups)
        self.mix = nn.Conv1d(dim, dim, kernel_size=1, groups=self.groups, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x to be 3-D (B, T, C), got {tuple(x.shape)}")
        batch, time, dim = x.shape

        y = self.mix(x.transpose(1, 2))
        y = y.reshape(batch, self.groups, dim // self.groups, time)
        y = y.transpose(1, 2).contiguous()
        y = y.reshape(batch, dim, time)
        return y.transpose(1, 2).contiguous()


class LightConvChannelMixer(nn.Module):
    """Ultra-light grouped channel mixer for sparse memory features."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = int(config["n_hidden_neurons"])
        groups = int(config.get("channel_mixer_groups", 4))
        self.norm = SequenceNorm(dim, _norm_type(config))
        self.mix = GroupedChannelMixer(
            dim,
            groups=groups,
            bias=bool(config.get("channel_mixer_light_bias", False)),
        )
        self.lif = BatchMajorLIF(config)

    def forward(self, x: Tensor) -> Tensor:
        return self.lif(self.mix(self.norm(x)))


class ConvNorm1d(nn.Module):
    """Batch-major Conv1d followed by GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.GroupNorm(_validate_group_count(out_channels, groups), out_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x.transpose(1, 2))
        y = self.norm(y)
        return y.transpose(1, 2).contiguous()


class BottleneckConvChannelMixer(nn.Module):
    """Bottleneck temporal Conv1d mixer for lighter cross-channel fusion."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = int(config["n_hidden_neurons"])
        ratio = int(config.get("channel_mixer_bottleneck_ratio", 4))
        bottleneck_dim = _validate_bottleneck_dim(dim, ratio)
        kernel_size = int(config.get("channel_mixer_kernel_size", 3))
        dilation = int(config.get("channel_mixer_dilation", 1))
        norm_groups = int(config.get("channel_mixer_norm_groups", 8))
        if kernel_size % 2 == 0:
            raise ValueError("channel_mixer_kernel_size must be odd")
        if dilation < 1:
            raise ValueError("channel_mixer_dilation must be >= 1")
        padding = dilation * (kernel_size // 2)

        self.reduce = ConvNorm1d(dim, bottleneck_dim, kernel_size=1, groups=norm_groups)
        self.reduce_lif = BatchMajorLIF(config)
        self.temporal = ConvNorm1d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=norm_groups,
        )
        self.temporal_lif = BatchMajorLIF(config)
        self.expand = ConvNorm1d(bottleneck_dim, dim, kernel_size=1, groups=norm_groups)
        self.output_lif = BatchMajorLIF(config)

    def forward(self, x: Tensor) -> Tensor:
        y = self.reduce_lif(self.reduce(x))
        y = self.temporal_lif(self.temporal(y))
        y = self.expand(y)
        return self.output_lif(y + x)


class MultiTauBlock(nn.Module):
    """Ultra-lightweight Multi-tau block based entirely on static internal fusion."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = int(config["n_hidden_neurons"])
        taus = config.get("taus", [5.0, 20.0, 100.0])

        self.context_norm = SequenceNorm(dim, _norm_type(config))

        self.memory = MultiTauLIFMemory(
            dim=dim,
            taus=taus,
            v_th=float(config.get("v_threshold", 1.0)),
            surrogate=config.get("surrogate", "atan"),
            learnable_alpha=bool(config.get("learnable_alpha", True)),
            mode=config.get("memory_mode", "soft"),
            fuse=True,
            branch_fusion_mode=config.get("branch_fusion_mode", "channel_adaptive"),
            detach_reset=bool(config.get("detach_reset", False)),
            decay_input=bool(config.get("decay_input", True)),
            surrogate_alpha=float(config.get("surrogate_alpha", 2.0)),
            backend=config.get("backend", "torch"),
        )

        mixer_type = config.get("channel_mixer_type", "linear")
        if mixer_type == "linear":
            self.channel_mixer = ChannelMixer(config)
        elif mixer_type == "conv":
            self.channel_mixer = BottleneckConvChannelMixer(config)
        elif mixer_type == "light-conv":
            self.channel_mixer = LightConvChannelMixer(config)
        elif mixer_type == "none":
            self.channel_mixer = nn.Identity()
        else:
            raise ValueError("channel_mixer_type must be one of: linear, conv, light-conv, none")
        self.dropout = nn.Dropout(float(config.get("dropout_l", 0.0))) if config.get("use_dp", False) else nn.Identity()

        # --- logging ---
        logger.info(f"channel_mixer_type: {mixer_type}")

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        residual = x

        y = self.context_norm(x)

        y, _ = self.memory(y)
        y = self.channel_mixer(y)
        y = self.dropout(y)

        y = _apply_attention_mask(y, attention_mask)

        return _apply_attention_mask(residual + y, attention_mask)


class MaskedTemporalPool(nn.Module):
    """Temporal mean pooling that ignores padded frames."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x to be 3-D (B, T, D), got {tuple(x.shape)}")
        if attention_mask is None:
            return x.mean(dim=1)
        mask = _normalize_attention_mask(attention_mask, x).to(dtype=x.dtype)
        denom = mask.sum(dim=1).clamp_min(self.eps)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)


class MultiTauLIFCommander(nn.Module):
    """Keyword-spotting model built from multi-tau LIF memory blocks."""

    def __init__(self, config: Mapping[str, Any] | DictConfig) -> None:
        super().__init__()
        self.config = _to_omegaconf(config)
        self.return_sequence_logits = bool(self.config.get("return_sequence_logits", True))
        self.frontend = self._build_frontend(self.config)
        self.blocks = nn.ModuleList(MultiTauBlock(self.config) for _ in range(int(self.config.get("depths", 1))))
        self.pool = MaskedTemporalPool()
        self.classifier = nn.Linear(
            int(self.config["n_hidden_neurons"]),
            int(self.config["n_outputs"]),
            bias=bool(self.config.get("head_bias", False)),
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        attention_mask = None if attention_mask is None else _normalize_attention_mask(attention_mask, x)
        x = self.frontend(x, attention_mask)
        for block in self.blocks:
            x = block(x, attention_mask)

        if self.return_sequence_logits:
            logits = self.classifier(_apply_attention_mask(x, attention_mask))
            return logits.transpose(0, 1).contiguous()

        pooled = self.pool(x, attention_mask)
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def branch_balance_loss(self) -> Tensor:
        """Average block-level branch-usage regularization loss."""
        losses = [block.memory.branch_balance_loss() for block in self.blocks]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return self.classifier.weight.new_zeros(())
        return torch.stack(losses).mean()

    def multi_tau_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Return trainable tau parameters and channel-fusion logits separately."""
        tau_parameters = []
        fusion_parameters = []
        for block in self.blocks:
            tau_parameters.extend(
                parameter for parameter in block.memory.alpha_parameters()
                if parameter.requires_grad
            )
            fusion_parameters.extend(
                parameter for parameter in block.memory.fusion_parameters()
                if parameter.requires_grad
            )
        return tau_parameters, fusion_parameters

    @staticmethod
    def _build_frontend(config: Mapping[str, Any]) -> nn.Module:
        frontend_type = config.get("frontend_type", "see_lite")
        if frontend_type == "see_lite":
            return SEELiteFrontend(config)
        if frontend_type == "linear":
            return LinearFrontend(config)
        raise ValueError("frontend_type must be either 'see_lite' or 'linear'")


def _to_omegaconf(config: Mapping[str, Any] | DictConfig) -> DictConfig:
    if OmegaConf.is_config(config):
        return config  # type: ignore[return-value]
    return OmegaConf.create(dict(config), flags={"allow_objects": True})


def _resolve_surrogate(config: Mapping[str, Any]) -> nn.Module:
    surrogate_name = config.get("surrogate", "atan")
    if surrogate_name != "atan":
        raise ValueError(f"Unsupported surrogate: {surrogate_name}")
    return surrogate.ATan(alpha=float(config.get("surrogate_alpha", 2.0)))


def _lif_v_reset(config: Mapping[str, Any]) -> float | None:
    if "v_reset" in config:
        value = config.get("v_reset")
        if isinstance(value, str) and value.lower() == "none":
            return None
        return value

    memory_mode = str(config.get("memory_mode", "soft")).lower()
    if memory_mode == "soft":
        return None
    return 0.0


def _norm_type(config: Mapping[str, Any]) -> str:
    return str(config.get("norm_type", "bn")).lower()


def _validate_group_count(num_channels: int, requested_groups: int) -> int:
    if requested_groups < 1:
        raise ValueError("group count must be >= 1")
    if num_channels % requested_groups != 0:
        raise ValueError(
            f"group count {requested_groups} must divide channel count {num_channels}; "
            "adjust channel_mixer_norm_groups or bottleneck ratio"
        )
    return requested_groups


def _validate_bottleneck_dim(dim: int, ratio: int) -> int:
    if ratio < 1:
        raise ValueError("channel_mixer_bottleneck_ratio must be >= 1")
    bottleneck_dim = dim // ratio
    if bottleneck_dim < 1:
        raise ValueError(
            f"channel_mixer_bottleneck_ratio={ratio} is too large for hidden dim {dim}; "
            "it would collapse the bottleneck to zero channels"
        )
    return bottleneck_dim


def _validate_input(x: Tensor, in_dim: int) -> None:
    if x.dim() != 3:
        raise ValueError(f"expected x to be 3-D (B, T, C), got {tuple(x.shape)}")
    if x.size(-1) != in_dim:
        raise ValueError(f"expected input dim {in_dim}, got {x.size(-1)}")


def _normalize_attention_mask(attention_mask: Tensor, x: Tensor) -> Tensor:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be shaped (B, T)")
    if tuple(attention_mask.shape) != tuple(x.shape[:2]):
        raise ValueError(f"expected attention_mask shape {tuple(x.shape[:2])}, got {tuple(attention_mask.shape)}")
    return attention_mask.to(device=x.device, dtype=torch.bool)


def _apply_attention_mask(x: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return x
    mask = _normalize_attention_mask(attention_mask, x)
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)
