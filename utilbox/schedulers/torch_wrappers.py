import torch


class CosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, batches_per_epoch: int, max_epochs: int, min_lr: float = 0):
        super(CosineAnnealingLR, self).__init__(optimizer, T_max=batches_per_epoch * max_epochs, eta_min=min_lr)
