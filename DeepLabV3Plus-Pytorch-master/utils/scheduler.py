from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

class PolyLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_iters: int, power: float = 0.9, min_lr: float = 1e-6, last_epoch: int = -1):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.max_iters:
            return [self.min_lr for _ in self.base_lrs]
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]