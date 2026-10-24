"""TdnnLstm ANN（KWS 教师/基线，约 32.1M 参数）。

结构（与历史 checkpoint 逐键兼容，输入 [N, 80, T] FBank → 输出 [N, C] logits）：
  Conv1d 80→768 k3 s1 → ReLU → BN(affine=False)
  Conv1d 768→768 k3 s1 → ReLU → BN(affine=False)
  Conv1d 768→768 k3 s{subsampling_factor} → ReLU → BN(affine=False)
  6 × (LSTM 768 → BN(affine=False) → Dropout(0.1) → residual)
  时间均值池化 → Linear 768→C

已注册进框架模型表（alpha/asr/models → conf.model.name=TdnnLstm 可直接路由），
加载 kd_teacher/tdnn_ann_29class.pt 直接兼容。
"""

import torch
import torch.nn as nn


class TdnnLstm(nn.Module):

    def __init__(self, num_features: int, num_classes: int, subsampling_factor: int = 3):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(num_features, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
            nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
            nn.Conv1d(768, 768, kernel_size=3, stride=self.subsampling_factor, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=768, hidden_size=768, num_layers=1) for _ in range(6)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=768, affine=False) for _ in range(6)]
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, self.num_classes)

    def forward(self, x):
        # x: [N, C, T] → [N, num_classes]
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(2, 0, 1)
            x_new = self.dropout(x_new)
            x = x_new + x
        x = x.transpose(1, 0)
        x = torch.mean(x, dim=1)
        return self.linear(x)
