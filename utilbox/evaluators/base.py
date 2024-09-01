from abc import ABC, abstractmethod
from typing import Dict


class BaseEvaluator(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs) -> (Dict, Dict):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError
