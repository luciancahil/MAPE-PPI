from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
import math

class CosSimScheduler(LRScheduler):
    cos_sim = None

    phi = (1 + math.sqrt(5)) / 2
    increase_gamma = 1.1
    decrease_gamma = 0.2
    loss_ratio = 1
    old_lr = None

    def __init__(self, optimizer: Optimizer, catastrophic_gamma = 0.001, increase_gamma=1.1, last_epoch=-1):
        super(CosSimScheduler, self).__init__(optimizer, last_epoch)

        self.optimizer = optimizer
        self.increase_gamma = increase_gamma
        self.catastrophic_gamma = catastrophic_gamma



    def cos_step(self, cos_sim=None, loss_ratio = 0, old_lr=None):
        self.cos_sim = cos_sim
        self.old_lr = old_lr
        self.loss_ratio = loss_ratio
        self.step()
        self.cos_sim = None
        self.loss_ratio = None
        self.old_lr = None


    def get_lr(self):
        
        if(self.old_lr is not None):
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            gamma = (self.old_lr / current_lr) * self.catastrophic_gamma
        elif(self.loss_ratio > 2):
            gamma = self.catastrophic_gamma
        elif(self.cos_sim is None):
            gamma = 1
        elif(self.cos_sim > -1.1):
            gamma = self.increase_gamma
        else:
            gamma = self.decrease_gamma

        return [group['lr'] * gamma for group in self.optimizer.param_groups]