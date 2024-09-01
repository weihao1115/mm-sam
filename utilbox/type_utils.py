import numpy as np
import torch


def to_numpy(input_tensor: torch.Tensor):
    if isinstance(input_tensor, np.ndarray):
        return input_tensor

    if hasattr(input_tensor, 'detach'):
        input_tensor = input_tensor.detach()
    if hasattr(input_tensor, 'cpu'):
        input_tensor = input_tensor.cpu()
    return input_tensor.numpy()


def str_is_float(string):
    if string.count('.') == 1:
        left, right = string.split('.')
        if left.isdigit() or (left.startswith('-') and left[1:].isdigit()):
            if right.isdigit():
                return True
    else:
        raise ValueError
