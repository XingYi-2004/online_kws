"""MultiTauLIFCommanderKWS：KWS 推理/在线适应形态（继承 MultiTauLIFCommander）。

与父模型是同种模型，仅两项适配：
  1. AvgPool 下采样（subsampling_factor，无参数，checkpoint 键不变）
  2. forward 输入为 FBank [N, 80, T] + lengths（内部转掩码后走父类前向）

用于在线 AdaKWS（examples/carKWS_online/local/train_adakws.py）、固定评测
（examples/carKWS_offline/local/eval_kws.py）与效率测量。conf 须与
online_adakws.yaml 的 model 块一致（含 return_sequence_logits: false）。
"""

import torch
import torch.nn as nn

from alpha.asr.model.multitau_lif_commander.multitau_lif_commander import (
    MultiTauLIFCommander,
)


def subsampling_mask(t_out: int, lengths: torch.Tensor, factor: int, device) -> torch.Tensor:
    """降采样后的有效帧掩码 [B, T_out]（lengths 为原始 FBank 帧数）。

    供本模型与离线 Lightning 系统（kws/system.MultiTauKwsKd）共用。
    """
    adj = torch.clamp((lengths + factor - 1) // factor, min=1, max=t_out)
    return torch.arange(t_out, device=device).unsqueeze(0) < adj.unsqueeze(1)


class MultiTauLIFCommanderKWS(MultiTauLIFCommander):

    def __init__(self, config):
        super().__init__(config)  # frontend / blocks / pool / classifier 全部继承
        ss = int(config.get("subsampling_factor", 2))
        self.subsample_factor = ss
        self.subsample = nn.AvgPool1d(kernel_size=ss, stride=ss) if ss > 1 else nn.Identity()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: FBank [N, 80, T]；lengths: 每句有效帧数 → 整句 logits [N, C]。"""
        x = self.subsample(x).transpose(1, 2)  # [N, T', F]
        attention_mask = None
        if lengths is not None:
            attention_mask = subsampling_mask(x.shape[1], lengths,
                                              self.subsample_factor, x.device)
        return super().forward(x, attention_mask)
