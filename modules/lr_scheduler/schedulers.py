import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        # warmup 
        if epoch < self.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # cosine decay
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
