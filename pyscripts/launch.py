#!/usr/bin/env python
from utilbox.global_config import PROJECT_ROOT, EXP_ROOT

import argparse
import os.path
import random
import sys
from os.path import join
from typing import List, Callable

from utilbox.parse_utils import str2list, str2bool, str2dict
from utilbox.train_managers.base import TrainManager, is_main_proc

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import GPUtil
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import parse as V
from GPUtil import GPU, getGPUs
from utilbox.yaml_utils import load_yaml


def get_idle_port() -> str:
    """
    find an idle port to used for distributed learning

    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt = str(random.randint(15000, 30000))
    if tt not in procarr:
        return tt
    else:
        return get_idle_port()


def get_idle_gpu(gpu_num: int = 1) -> List[GPU]:
    """find idle GPUs for distributed learning."""
    sorted_gpus = sorted(getGPUs(), key=lambda g: g.memoryUtil)
    if len(sorted_gpus) < gpu_num:
        raise RuntimeError(
            f"Your machine doesn't have enough GPUs ({len(sorted_gpus)}) as you specified ({gpu_num})!")
    sorted_gpus = sorted_gpus[:gpu_num]
    return sorted_gpus


def set_random_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        # For more details about 'CUBLAS_WORKSPACE_CONFIG',
        # please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        if V(torch.version.cuda) >= V("10.2"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # warn_only argument is added to use_deterministic_algorithms() after 1.11.0
        # for more information please refer to https://github.com/pytorch/pytorch/issues/64883
        if V(torch.__version__) >= V("1.11"):
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(mode=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(args: argparse.Namespace, device: torch.device, rank_on_node: int):
    """create a training manager to run the experiment (train or test)"""
    train_manager = TrainManager(local_rank=rank_on_node, device=device, args=args)
    if not args.test_only:
        if is_main_proc(): print("Start training!")
        train_manager.train()
        if is_main_proc(): print("Finish training!")
    # ensure that the subsequent testing after DDP training is only conducted on the main process (rank0)
    if not args.train_only and is_main_proc():
        if is_main_proc(): print("Start testing!")
        train_manager.test()
        if is_main_proc(): print("Finish testing!")


class Launcher:
    worker_fn: Callable = run_experiment

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        group = parser.add_argument_group("Shared Arguments")
        group.add_argument('--config_name', default=None, type=str)
        group.add_argument('--seed', default=3407, type=int)
        group.add_argument('--gpu_num', default=1, type=int)
        group.add_argument('--world_size', default=None, type=int)
        group.add_argument('--node_first_rank', default=0, type=int)
        group.add_argument('--best_model_selection', default=None, type=str2list)
        group.add_argument('--use_amp', default=False, type=str2bool)
        group.add_argument('--train_only', default=False, type=str2bool)
        group.add_argument('--test_only', default=False, type=str2bool)
        group.add_argument('--train_agent', default=None, type=str)
        group.add_argument('--agent_kwargs', default={}, type=str2dict)

        group = parser.add_argument_group("Training Arguments")
        group.add_argument('--train_bs', default=None, type=int)
        group.add_argument('--train_workers_per_bs', default=0.5, type=float)
        group.add_argument('--train_epoch_num', default=None, type=int)
        group.add_argument('--valid_bs', default=1, type=int)
        group.add_argument('--valid_workers_per_bs', default=0.5, type=float)

        group = parser.add_argument_group("Testing Arguments")
        group.add_argument('--ckpt_path', type=str, default='checkpoints/')
        group.add_argument('--test_bs', default=1, type=int)
        group.add_argument('--test_workers_per_bs', default=0.5, type=float)
        return parser

    @classmethod
    def configure(cls) -> argparse.Namespace:
        args = cls.get_parser().parse_args()
        assert PROJECT_ROOT is not None, "Please register PROJECT_ROOT in utilbox/global_config.py!"
        config_yaml_path = f"{PROJECT_ROOT}/config/{args.config_name}"
        if not config_yaml_path.endswith('.yaml'):
            config_yaml_path += '.yaml'
        config = load_yaml(config_yaml_path)

        known_args_list = [item.dest for item in cls.get_parser()._actions]
        for k, v in config.items():
            if k in known_args_list:
                setattr(args, k, v)
            else:
                raise ValueError(f"Unknown argument '{k}' in your .yaml file!")

        launch_name_list = args.config_name.split('/')
        if launch_name_list[-1].endswith('.yaml'):
            launch_name_list[-1] = launch_name_list[-1][:-len('.yaml')]
        launch_name = '/'.join(launch_name_list)
        args.launch_folder_path = join(EXP_ROOT, launch_name)
        os.makedirs(args.launch_folder_path, exist_ok=True)

        if sum([args.train_only, args.test_only]) > 1:
            raise ValueError(
                'Cannot set --train_only true and --test_only true at the same time! One of them should be False.'
            )

        # GPU number is forced to be 1 if no training is specified
        if args.test_only:
            args.gpu_num = 1
        if args.train_agent is None:
            raise ValueError("--train_agent must be specified!")
        if args.train_only and args.train_epoch_num is None:
            raise ValueError("--train_epoch_num must be specified for training!")

        if args.world_size is None:
            args.world_size = args.gpu_num

        # Initialize the GPU information from command context (higher priority)
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            all_gpus = GPUtil.getGPUs()
            used_gpu = [all_gpus[int(gpu_idx)] for gpu_idx in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            if len(used_gpu) < args.gpu_num:
                raise RuntimeError(
                    f"Meet different GPU configurations between --gpu_num ({args.gpu_num}) and "
                    f"CUDA_VISIBLE_DEVICES ({os.environ['CUDA_VISIBLE_DEVICES']})!"
                )
            else:
                used_gpu = used_gpu[:args.gpu_num]
        # Initialize the GPU automatically
        elif args.gpu_num > 0:
            used_gpu = get_idle_gpu(gpu_num=args.gpu_num)
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu.id) for gpu in used_gpu])
        else:
            used_gpu = []

        args.used_gpu = used_gpu
        return args

    @classmethod
    def launch(cls):
        # Get the Command Line Arguments
        args = cls.configure()

        # launch the experiment process for both single-GPU and multi-GPU settings
        if args.gpu_num <= 1:
            cls.launch_worker(worker_id=0, args=args)
        else:
            # initialize multiprocessing start method
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                try:
                    mp.set_start_method('forkserver')
                    print("Fail to initialize multiprocessing module by spawn method. "
                          "Use forkserver method instead. Please be careful about it.")
                except RuntimeError as e:
                    raise RuntimeError(
                        "Your server supports neither spawn or forkserver method as multiprocessing start methods. "
                        f"The error details are: {e}"
                    )

            # dist_url is fixed to localhost here, so only single-node DDP is supported now.
            args.dist_url = "tcp://127.0.0.1" + f':{get_idle_port()}'
            # spawn one subprocess for each GPU
            mp.spawn(cls.launch_worker, nprocs=args.gpu_num, args=(args,))

        # exit the program at the end of the job
        sys.exit(0)

    @classmethod
    def launch_worker(cls, worker_id, args):
        # reproduction setting
        set_random_seeds(args.seed)

        # initialize the GPU devices and do some preparatory works
        if isinstance(worker_id, str):
            worker_id = int(worker_id)

        # Initialize DDP for multi-GPU setting
        if args.gpu_num > 1:
            dist.init_process_group(
                backend='nccl', init_method=args.dist_url,
                world_size=args.world_size, rank=args.node_first_rank + worker_id
            )

        # Initialize the used GPU for the current process
        if args.gpu_num > 0:
            device = torch.device(f"cuda:{worker_id}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # set matrix calculate precision (shared by all subprocesses)
        torch.set_float32_matmul_precision('medium')

        try:
            cls.worker_fn(args, device, worker_id)
        except KeyboardInterrupt:
            print("Catch a KeyBoardInterrupt! Ready to exit the program...")
        # the following two cases are used to properly terminate non-0 ranks in the DDP mode
        except RuntimeError as e:
            if dist.is_initialized() and "Connection reset by peer" in str(e):
                print(f"Catch a RuntimeError at the rank{dist.get_rank()}: {e}!")
            # re-raise the exceptions not caused by DDP
            else:
                raise e
        except Exception as e:
            if V(torch.__version__) >= V('2.0.0') and isinstance(e, dist.DistBackendError):
                print(f"Catch a DistBackendError at the rank{dist.get_rank()}: {e}")
            else:
                raise e

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    Launcher.launch()
