"""KWS 推理/在线适应模型：AvgPool 下采样 + MultiTauLIFCommander。

架构:
  FBank (B, 80, T) → subsample(AvgPool) → MultiTauLIFCommander → 掩码池化
  → Linear(C) → 类别 logits

供在线 AdaKWS（examples/carKWS_online/local/train_adakws.py）与固定评测
（examples/carKWS_offline/local/eval_kws.py）使用；离线 Lightning 系统
（system.MultiTauKwsKd）的推理路径与其等价（键 model.* 与 backbone.* 由
load_pretrained_model 的映射打通）。
"""

import torch
import torch.nn as nn

from alpha.asr.model.multitau_lif_commander.multitau_lif_commander import (
    MultiTauLIFCommander,
)


def subsampling_mask(t_out: int, lengths: torch.Tensor, factor: int, device) -> torch.Tensor:
    """降采样后的有效帧掩码 [B, T_out]（lengths 为原始 FBank 帧数）。

    供本模块与离线 Lightning 系统（system.MultiTauKwsKd）共用。
    """
    adj = torch.clamp((lengths + factor - 1) // factor, min=1, max=t_out)
    return torch.arange(t_out, device=device).unsqueeze(0) < adj.unsqueeze(1)


class MultiTauLIFCommanderKWS(nn.Module):
    """subsample + MultiTauLIFCommander（参数键：subsample 无参数，主干 backbone.*）。

    config 与 online_adakws.yaml 的 model 块一致（subsampling_factor 控制下采样）。
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = dict(config)
        cfg["return_sequence_logits"] = False  # 固定整句 logits 路径

        ss = int(cfg.get("subsampling_factor", 2))
        self.subsample_factor = ss
        self.subsample = nn.AvgPool1d(kernel_size=ss, stride=ss) if ss > 1 else nn.Identity()
        self.backbone = MultiTauLIFCommander(cfg)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: FBank [N, 80, T]；lengths: 每句有效帧数 → logits [N, C]。"""
        x = self.subsample(x).transpose(1, 2)  # [N, T', F]
        attention_mask = None
        if lengths is not None:
            attention_mask = subsampling_mask(x.shape[1], lengths,
                                              self.subsample_factor, x.device)
        return self.backbone(x, attention_mask)
