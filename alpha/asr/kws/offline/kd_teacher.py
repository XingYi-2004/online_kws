"""KD 教师模型：TdnnLSTM ANN（32.1M 参数，29 类中文车载指令）。

来源：tdnn_lstm_6ann_768_new2/exp/pretrained.pt（兄弟项目 SNN/ 训练产物，
      KD-MT-08 离线蒸馏时的教师，权重副本固化于 kd_teacher/tdnn_ann_29class.pt）。
训练配置（由 KD-MT-08 训练日志还原）：alpha=0.7，T=10.0。
输入 [N, 80, T] FBank → 输出 [N, 29] logits（内部自带 3 倍下采样）。
"""

import torch
import torch.nn as nn


class TdnnLstm(nn.Module):
    """TdnnLSTM ANN（与 SNN/model_ann.py 保持一致，供 KD 教师加载）。"""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T] → [N, 29]
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
