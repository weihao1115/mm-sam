from abc import abstractmethod, ABC
from typing import Dict

import torch


class OptimAgent(ABC):

    def __init__(self, accum_grad: int = 1, use_amp: bool = False):
        # initialize the general part of the scheduler
        self.distributed = torch.distributed.is_initialized()
        self.accum_grad = accum_grad
        self.use_amp = use_amp

    def step(self, losses: Dict[str, torch.Tensor], total_step_num: int):
        # back-propagate the loss
        self.loss_backward(losses)

        # update the model parameters if the accumulation interval is met
        if total_step_num % self.accum_grad == 0:
            self.model_optimization()

    @abstractmethod
    def loss_backward(self, losses: Dict):
        raise NotImplementedError

    @abstractmethod
    def model_optimization(self):
        raise NotImplementedError

    @abstractmethod
    def get_optim_lr(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        raise NotImplementedError
