import os
import random
from functools import partial
from typing import Callable, Sequence, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def batch_list_to_dict(batch_list: Sequence[Dict]) -> Dict[str, List]:
    # preprocess List[Dict[str, Any]] to Dict[str, List[Any]]
    batch_dict = dict()
    while len(batch_list) != 0:
        ele_dict = batch_list[0]
        if ele_dict is not None:
            for key in ele_dict.keys():
                if key not in batch_dict.keys():
                    batch_dict[key] = []
                batch_dict[key].append(ele_dict[key])
        # remove the redundant data for memory safety
        batch_list.remove(ele_dict)
    return batch_dict


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.

    """
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_dataloader(
        dataset: torch.utils.data.Dataset,
        worker_num: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        distributed: bool = False,
        worker_seed: int = 3407,
        same_worker_seed: bool = False,
) -> DataLoader:
    sampler = None
    if distributed:
        world_size = torch.distributed.get_world_size()
        assert batch_size % world_size == 0, (
            "Batch size must be divisible by the world size. "
            f"You have world size {world_size} but batch size {batch_size}!"
        )
        batch_size = int(batch_size / world_size)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        sampler.set_epoch(epoch=0)

    if collate_fn is None:
        assert hasattr(dataset, "collate_fn")
        collate_fn = dataset.collate_fn

    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        num_workers=worker_num, sampler=sampler, pin_memory=pin_memory,
        drop_last=drop_last, collate_fn=collate_fn,
        worker_init_fn=partial(
            # use different seeds for each epoch to introduce more randomness
            worker_init_fn, base_seed=worker_seed, same_worker_seed=same_worker_seed
        )
    )
