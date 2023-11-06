import warnings

from torch.optim.lr_scheduler import LRScheduler


class InverseSqrtLR(LRScheduler):
    def __init__(self, optimizer, lr, warmup_steps, pre_warmup_lr=None, last_epoch=-1, verbose=False):
        warmup_steps = max(warmup_steps, 1)
        self.lr = lr * warmup_steps**0.5
        self.warmup_steps = warmup_steps
        self.pre_warmup_lr = pre_warmup_lr if pre_warmup_lr is not None else lr
        super().__init__(optimizer, last_epoch, verbose)

    def _process_lr(self, _):
        warmup_factor = min(self.last_epoch/self.warmup_steps, 1) # this grows linearly from 0 to 1 during the warmup
        base_lr = self.lr / max(self.last_epoch, self.warmup_steps)**0.5
        return warmup_factor * base_lr + (1-warmup_factor)*self.pre_warmup_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        lr = self._process_lr(self.lr)
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self._process_lr(base_lr) for base_lr in self.base_lrs]