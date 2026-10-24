"""KWS 自定义 LR 调度器。

LinearWarmupCosineLR 逐位复刻原版 ANN 教师调度（icefall SequentialLR 组合）：
  LinearLR(start_factor=0.1, end_factor=1.0, total_iters=3)
  + CosineAnnealingLR(T_max=num_epochs-3, eta_min=1e-6)
  经 SequentialLR(milestones=[3]) 按 epoch 步进。
"""

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


class LinearWarmupCosineLR(SequentialLR):

    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 warmup_start_factor=0.1, eta_min=0.0):
        warmup = LinearLR(optimizer, start_factor=warmup_start_factor,
                          end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs,
                                   eta_min=eta_min)
        super().__init__(optimizer, [warmup, cosine], milestones=[warmup_epochs])
