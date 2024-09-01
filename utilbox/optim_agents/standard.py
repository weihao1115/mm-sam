from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler

from packaging.version import parse as V
# After torch V2.0, the base class has been renamed to LRScheduler.
# But _LRScheduler is still available for compatibility with previous versions
if V(torch.__version__) >= V('2.0'):
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from utilbox.optim_agents.base import OptimAgent


class StandardOptimAgent(OptimAgent):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 # super arguments
                 accum_grad: int = 1,
                 use_amp: bool = False,
                 # customized arguments
                 scheduler: Optional[LRScheduler] = None,
                 grad_clip: float = None,
                 grad_norm_type: float = 2.0):

        super(StandardOptimAgent, self).__init__(accum_grad=accum_grad, use_amp=use_amp)

        self.grad_clip = grad_clip
        self.grad_norm_type = grad_norm_type

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Initialize the gradient scaler for AMP training
        self.scaler = None
        if self.use_amp:
            self.scaler = GradScaler()

    def loss_backward(self, losses: Dict):
        assert 'loss' in losses.keys(), "Please give your target loss by an item named 'loss'!"
        loss = losses['loss']
        # average the loss for accumulation
        loss /= self.accum_grad
        # backward the loss in either the amp mode or the normal mode
        self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward()

    def model_optimization(self):
        # unscale the gradients in advance to enable gradient clipping in the amp setting
        # refer: https://pytorch.org/docs/1.10/notes/amp_examples.html#working-with-unscaled-gradients
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = None
        if self.grad_clip is not None:
            # apply the gradient clipping right before updating the target parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=self.optimizer.param_groups[0]['params'],
                max_norm=self.grad_clip, norm_type=self.grad_norm_type
            )

        # optimize the target parameters only when the values of gradients are not infinite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            if self.scaler is not None:
                # self.optimizer.step() will be skipped if grad_norm is NaN or inf in self.scaler.step(self.optimizer)
                # so self.scheduler.step() is not called in this case
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                # if scaler increase after update(), self.optimizer.step() will be skipped,
                # so self.scheduler.step() should not be called.
                # ref: https://github.com/Pointcept/Pointcept/blob/9a3f603d2cb5473df728296964abcde7aa94c8c9/pointcept/engines/train.py#L189
                scaler = self.scaler.get_scale()
                self.scaler.update()
                if self.scheduler is not None and scaler <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        # Turn the gradients of the target parameters of this optimizer to zero at the end to allow grad accum
        self.optimizer.zero_grad()

    def get_optim_lr(self) -> Dict[str, float]:
        return {'main_opt': self.optimizer.param_groups[0]['lr']}

    def state_dict(self) -> Dict:
        return dict(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            scaler=None if self.scaler is None else self.scaler.state_dict()
        )

    def load_state_dict(self, state_dict: Dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
