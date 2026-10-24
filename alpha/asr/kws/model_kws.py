"""在线学习模型：MultiTauLIFCommander 封装（AdaKWS 测试时适应用）。

架构:
  FBank (B, 80, T) → subsample → MultiTauLIFCommander → 掩码池化 → Linear(C)
  → 类别 logits

用法:
  model = MultiTauLIFCommanderKWS(config)    # config 来自 da_aol.yaml 的 model 块
  class_logits = model(features, lengths)

历史说明: 对抗域适应组件（GRL / CDAN RandomLayer / SNN·ANN 域判别器）已随
DA 系列实验清理；旧 checkpoint（如 90.10% 冠军）中的 domain_classifier.* 键
由 utils/train_utils.load_pretrained_model 的键映射自动丢弃，可直接加载。
"""

import torch
import torch.nn as nn

from alpha.asr.model.multitau_lif_commander.multitau_lif_commander import (
    MultiTauLIFCommander,
    _normalize_attention_mask,
)


class MultiTauLIFCommanderKWS(nn.Module):
    """MultiTauLIFCommander 封装：AvgPool 下采样 + 掩码前向。

    原名 MultiTauLIFCommanderDA（DA 实验时代命名）；域判别器已清理，
    本体 = subsample + MultiTauLIFCommander（frontend → blocks → pool → classifier）。

    Config 说明（与 da_aol.yaml 的 model 块一致）：
      n_inputs: 80           FBank 维度
      n_hidden_neurons: 384  隐藏维度
      n_outputs: 29          类别数
      depths: 5              MultiTauBlock 层数
      taus: [2, 4, 8]        多时间常数
      subsampling_factor: 2  降采样步长
    """

    def __init__(self, config: dict):
        super().__init__()

        # ---- 特征提取器：MultiTauLIFCommander ----
        _cfg = dict(config)
        # 固定走 pool → classifier 路径，输出整句 logits
        _cfg["return_sequence_logits"] = False

        # 确保 MT 所需的关键配置（da_aol.yaml 已含大部分，这里兜底）
        _cfg.setdefault("frontend_type", "see_lite")
        _cfg.setdefault("taus", [2.0, 4.0, 8.0])
        _cfg.setdefault("learnable_alpha", False)
        _cfg.setdefault("memory_mode", "soft")
        _cfg.setdefault("branch_fusion_mode", "channel_adaptive")
        _cfg.setdefault("channel_mixer_type", "conv")
        _cfg.setdefault("norm_type", "bn")
        _cfg.setdefault("surrogate", "atan")
        _cfg.setdefault("surrogate_alpha", 2.0)
        _cfg.setdefault("decay_input", True)
        _cfg.setdefault("head_bias", False)
        _cfg.setdefault("use_dp", True)
        _cfg.setdefault("dropout_l", 0.15)

        # 降采样（与离线训练的 AlphaSNNForKWS 一致）
        ss = int(_cfg.get("subsampling_factor", 2))
        self.subsample_factor = ss
        self.subsample = nn.AvgPool1d(kernel_size=ss, stride=ss) if ss > 1 else nn.Identity()

        # 核心特征提取器（含 frontend / blocks / pool / classifier）
        self.backbone = MultiTauLIFCommander(_cfg)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
          x:       FBank 特征，形状 [N, 80, T]
          lengths: 每句有效帧数

        Returns:
          class_logits [N, C]
        """
        # 第1步：降采样 [N, 80, T] → [N, T//ss, 80]
        x = self.subsample(x).transpose(1, 2)

        # 第2步：构建 attention_mask（屏蔽补零帧；lengths 为原始帧数，需换算到降采样后）
        attention_mask = None
        if lengths is not None:
            _, T_out = x.shape[0], x.shape[1]
            ss = self.subsample_factor
            adj_lengths = torch.clamp((lengths + ss - 1) // ss, min=1, max=T_out)
            attention_mask = torch.arange(T_out, device=x.device).unsqueeze(0) < adj_lengths.unsqueeze(1)

        # 第3步：手动跑主干（与 backbone(x, mask) 等价，保留部件级结构便于对照）
        mask = None if attention_mask is None else _normalize_attention_mask(attention_mask, x)
        h = self.backbone.frontend(x, mask)
        for block in self.backbone.blocks:
            h = block(h, mask)

        # 第4步：掩码池化（只平均有效帧）+ 分类头
        pooled = self.backbone.pool(h, mask)
        return self.backbone.classifier(pooled)
