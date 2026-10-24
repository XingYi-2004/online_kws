# SNN for speech command classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import rnn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class TdnnLstm(nn.Module):
    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        """
        Args:
          num_features:
            The input dimension of the model.
          num_classes:
            The output dimension of the model.
          subsampling_factor:
            It reduces the number of output frames by this factor.
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        # —— 当前结构: Conv → ReLU → BN（兼容旧检查点） ——
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=768,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
            nn.Conv1d(in_channels=768, out_channels=768,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
            nn.Conv1d(in_channels=768, out_channels=768,
                      kernel_size=3, stride=self.subsampling_factor, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=768, affine=False),
        )
        # —— 新顺序 Conv → BN → ReLU（注释保留，需要重新训练才能用） ——
        # self.tdnn = nn.Sequential(
        #     nn.Conv1d(in_channels=num_features, out_channels=768,
        #               kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(num_features=768, affine=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=768, out_channels=768,
        #               kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(num_features=768, affine=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=768, out_channels=768,
        #               kernel_size=3, stride=self.subsampling_factor, padding=1),
        #     nn.BatchNorm1d(num_features=768, affine=False),
        #     nn.ReLU(inplace=True),
        # )
        #ANN LSTM
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=768, hidden_size=768, num_layers=1) for _ in range(6)]
        )
        #ANN BatchNorm
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=768, affine=False) for _ in range(6)]
        )
        # self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        #ANN linear layer
        self.linear = nn.Linear(in_features=768, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            Its shape is [N, C, T]

        Returns:
          The output tensor has shape [N, T, C]
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (N, C, T) -> (T, N, C) -> how LSTM expects it

        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, N, C) -> (N, C, T) -> (T, N, C)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections

        x = x.transpose(
            1, 0
        )  # (T, N, C) -> (N, T, C) -> linear expects "features" in the last dim
        x = torch.mean(x, dim=1) # (N, T, C) -> (N, C)
        x = self.linear(x)
        return x
