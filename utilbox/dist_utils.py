import torch


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    else:
        return torch.distributed.get_world_size()


def get_global_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    else:
        return torch.distributed.get_rank()
