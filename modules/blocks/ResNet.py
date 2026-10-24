# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" ResNet implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    Reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pooling_layers


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        m_channels=32,
        feat_dim=40,
        embedding_size=192,
        pooling_func="TSTP",
        two_emb_layer=True,
        **kwargs
    ):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

        self.n_stats = 1 if pooling_func == "TAP" or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion
        )
        if feat_dim == 257:
            linear_in = 16896 
        else:
            linear_in = self.stats_dim * block.expansion * self.n_stats        
        self.seg_1 = nn.Linear(linear_in, embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)
        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.create({"feat_dim": 257})
    conf.merge_with_cli()
    x = torch.zeros(10, 1, conf.feat_dim, 300)  # B, 1, F, T
    model = ResNet(feat_dim=conf.feat_dim, embedding_size=192, pooling_func="TSTP")
    model.eval()
    out = model(x)
    print(out.shape)  # torch.Size([10, 192])

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))  # 6.34M
