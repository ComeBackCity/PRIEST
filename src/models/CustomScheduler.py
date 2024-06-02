import torch
import math
import numpy as np
from typing import Optional

class LinearIncreseWithReduceLRonPlateau(object):
    def __init__(self, 
        optimizer : torch.optim.Optimizer,
        epochs : int,
        base_lr : np.float32 = 0.0,
        max_lr : np.float32 = 1e-4,
        reduce_lr_on_plateau : bool = False,
        patience : int = 5,
        cooldown : int = 3,
        threshold : np.float32 = 1e-4
    ) -> None:
        self._optim = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.patience = patience
        self.cooldown = cooldown
        self.threshold = threshold
        self.gradient = (max_lr - base_lr) / ( epochs - 1)
        self.lrs = [base_lr]
        self.last_epoch  = 0


    def step(self, parameter: Optional[np.float32]):
        new_lr = self.base_lr + self.gradient * (self.last_epoch+1)
        self.lrs.append(new_lr)
        self.last_epoch += 1
        self._optim.param_groups[0]['lr'] = new_lr
        
