import argparse

import humanfriendly
import math
import os
from os.path import join

import numpy as np
import torch

from typing import List, Dict, Union, Callable, Iterator, Sequence, Mapping
from torch.cuda.amp import autocast
from tqdm import tqdm

from utilbox.dict_utils import average_dicts, merge_flatten_dict
from utilbox.file_utils import search_file_in_subfolder
from utilbox.log_utils import dict_to_log_message
from utilbox.train_agents.base import TrainAgent
from utilbox.import_utils import import_class


def is_main_proc():
    if not torch.distributed.is_initialized():
        return True
    else:
        rank = torch.distributed.get_rank()
        if rank != 0:
            return False
        if rank == 0:
            return True


def batch_to_cuda(batch: Dict, device: Union[str, torch.device], non_blocking: bool = False) -> Dict:
    if isinstance(device, str):
        device = torch.device(device)

    def data_to_cuda(input_data):
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
            return input_data.to(device=device, dtype=input_data.dtype, non_blocking=non_blocking)
        elif isinstance(input_data, torch.Tensor):
            return input_data.to(device=device, dtype=input_data.dtype, non_blocking=non_blocking)
        # Note: str should appear earlier than Sequence since str is also a kind of Sequence and it will cause
        # infinite iterative error
        elif isinstance(input_data, str) or input_data is None:
            return input_data
        elif isinstance(input_data, Sequence):
            return [data_to_cuda(item) for item in input_data]
        elif isinstance(input_data, Mapping):
            return {data_key: data_to_cuda(data_value) for data_key, data_value in input_data.items()}
        elif isinstance(input_data, int):
            return torch.LongTensor([input_data]).to(device=device, non_blocking=non_blocking)
        elif isinstance(input_data, float):
            return torch.FloatTensor([input_data]).to(device=device, non_blocking=non_blocking)
        elif isinstance(input_data, bool):
            return torch.BoolTensor([input_data]).to(device=device, non_blocking=non_blocking)
        else:
            raise TypeError(f"Unsupported data type: {type(input_data)}!")

    return data_to_cuda(batch)


def get_ckpt_by_path(ckpt_path: str) -> List[str]:
    if os.path.isdir(ckpt_path):
        checkpoint_list = search_file_in_subfolder(
            ckpt_path, return_name=False,
            tgt_match_fn=lambda file: (file.endswith('.pth') or file.endswith('.pt')) and file != 'resume.pth'
        )
    elif ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt'):
        checkpoint_list = [ckpt_path]
    else:
        raise RuntimeError(f'ckpt_path {ckpt_path} not existed!')
    checkpoint_list = sorted(checkpoint_list)
    return checkpoint_list


class TrainManager:

    def __init__(self, local_rank: int, device: torch.device, args: argparse.Namespace):
        # Initialize the General Variables for later usage
        self.distributed = torch.distributed.is_initialized()
        self.local_rank = local_rank
        self.device = device
        self.args = args
        self.launch_folder_path = args.launch_folder_path

        # Initialize the Model
        self.use_amp = self.args.use_amp
        self.train_epoch_num = self.args.train_epoch_num
        assert hasattr(args, 'train_agent')
        self.train_agent: TrainAgent = import_class(args.train_agent)(
            seed=args.seed, device=self.device, local_rank=self.local_rank,
            find_unused_parameters=False, use_amp=self.use_amp,
            train_only=args.train_only, test_only=args.test_only,
            train_bs=args.train_bs, train_workers_per_bs=args.train_workers_per_bs,
            train_epoch_num=self.train_epoch_num, pin_memory=True,
            valid_bs=args.valid_bs, valid_workers_per_bs=args.valid_workers_per_bs,
            test_bs=args.test_bs, test_workers_per_bs=args.test_workers_per_bs,
            folder_path=self.launch_folder_path, **args.agent_kwargs
        )
        if not isinstance(self.train_agent, TrainAgent):
            raise TypeError("Your training agent must be an instance of utilbox.train_agents.base.TrainAgent!")
        self.train_batch_num = self.train_agent.train_batch_num
        self.valid_batch_num = self.train_agent.valid_batch_num
        self.test_batch_num = self.train_agent.test_batch_num

        if not args.test_only:
            self.best_model_selection, self.best_model_performance = None, None
            if hasattr(args, 'best_model_selection') and args.best_model_selection is not None:
                self.best_model_selection = args.best_model_selection
                if not isinstance(self.best_model_selection[0][0], List):
                    self.best_model_selection = [self.best_model_selection]
                for item in self.best_model_selection:
                    assert len(item) in [2, 3] and item[1] in ['max', 'min']
                    if not isinstance(item[0], List):
                        item[0] = [item[0]]

                self.best_model_performance = {}
                for item in self.best_model_selection:
                    assert len(item) in [2, 3] and item[1] in ['min', 'max']
                    if len(item) == 2:
                        item.append(1)
                    metric_keys, metric_mode, model_num = item

                    best_dict_pointer = self.best_model_performance
                    for i in range(len(metric_keys)):
                        if metric_keys[i] not in best_dict_pointer.keys():
                            if i < len(metric_keys) - 1:
                                best_dict_pointer[metric_keys[i]] = {}
                                best_dict_pointer = best_dict_pointer[metric_keys[i]]
                            else:
                                best_dict_pointer[metric_keys[i]] = [
                                    {
                                        'metric_value': math.inf if metric_mode == 'min' else -math.inf,
                                        'ckpt_path': None
                                    }
                                    for _ in range(model_num)
                                ]
                        else:
                            best_dict_pointer = best_dict_pointer[metric_keys[i]]

    def batch_preprocess(self, batch: Dict):
        # non_blocking is activated if pin_memory is set to True
        return batch_to_cuda(batch, self.device, non_blocking=True)

    def save_checkpoint(self, saved_ckpt_path):
        if not saved_ckpt_path.endswith('.pth'):
            saved_ckpt_path += '.pth'
        saved_ckpt_path = saved_ckpt_path.replace(' ', '_')

        ckpt_pardir = os.path.dirname(saved_ckpt_path)
        os.makedirs(ckpt_pardir, exist_ok=True)
        torch.save(self.train_agent.state_dict()["model"], saved_ckpt_path)

    def save_best_models(self, valid_results: Dict, ckpt_prefix: str):
        assert self.best_model_selection is not None

        for metric_keys, metric_mode, model_num in self.best_model_selection:
            valid_dict_pointer = valid_results
            best_dict_pointer = self.best_model_performance
            for i in range(len(metric_keys) - 1):
                valid_dict_pointer = valid_dict_pointer[metric_keys[i]]
                best_dict_pointer = best_dict_pointer[metric_keys[i]]

            if metric_mode == 'max':
                is_better_fn = lambda x, y: x > y
                worst_fn = min
            elif metric_mode == 'min':
                is_better_fn = lambda x, y: x < y
                worst_fn = max
            else:
                raise RuntimeError

            curr_metric_value = valid_dict_pointer[metric_keys[-1]]
            curr_metric_dict_pointer = best_dict_pointer[metric_keys[-1]]
            if None in [item['ckpt_path'] for item in curr_metric_dict_pointer]:
                tgt_idx = 0
                while curr_metric_dict_pointer[tgt_idx]['ckpt_path'] is not None:
                    tgt_idx += 1
            else:
                curr_worst_metric = worst_fn([item['metric_value'] for item in curr_metric_dict_pointer])
                if is_better_fn(curr_metric_value, curr_worst_metric):
                    tgt_idx = 0
                    while curr_metric_dict_pointer[tgt_idx]['metric_value'] != curr_worst_metric:
                        tgt_idx += 1
                else:
                    continue

            tgt_dict = curr_metric_dict_pointer[tgt_idx]
            # make sure the old checkpoint is properly deleted
            old_ckpt_path = tgt_dict['ckpt_path']
            while old_ckpt_path is not None and os.path.exists(old_ckpt_path):
                os.remove(old_ckpt_path)

            # register the newly-added checkpoint
            tgt_dict['metric_value'] = curr_metric_value
            tgt_dict['ckpt_path'] = join(
                self.launch_folder_path, 'checkpoints', f'best {" ".join(metric_keys)} models',
                f'{ckpt_prefix}_metric={curr_metric_value:.4f}.pth'
            )
            tgt_dict['ckpt_path'] = tgt_dict['ckpt_path'].replace(' ', '_')

        # We save the best checkpoints to make sure self.state_dict() has registered all the best checkpoints
        for metric_keys, metric_mode, model_num in self.best_model_selection:
            best_dict_pointer = self.best_model_performance
            valid_dict_pointer = valid_results
            for i, m_key in enumerate(metric_keys):
                best_dict_pointer = best_dict_pointer[m_key]
                if i < len(metric_keys) - 1:
                    valid_dict_pointer = valid_dict_pointer[m_key]

            for item in best_dict_pointer:
                while item['ckpt_path'] is not None and not os.path.exists(item['ckpt_path']):
                    self.save_checkpoint(saved_ckpt_path=item['ckpt_path'])

            if metric_mode == 'max':
                best_fn = max
            elif metric_mode == 'min':
                best_fn = min
            else:
                raise RuntimeError
            # ensure there is no blank in the path of best checkpoint files
            best_metric_key = 'Best ' + metric_keys[-1]
            valid_dict_pointer[best_metric_key] = best_fn([item['metric_value'] for item in best_dict_pointer])

    def train(self):
        self.train_agent.train()
        for epoch in range(1, self.train_epoch_num + 1):
            train_pbar = None
            if is_main_proc():
                train_pbar = tqdm(total=self.train_batch_num, desc='train', leave=False)

            # Hook: something to be done before each training epoch
            self.train_agent.before_train_epoch()
            if self.distributed:
                assert hasattr(self.train_agent.train_loader.sampler, "set_epoch")
                self.train_agent.train_loader.sampler.set_epoch(epoch)
            train_iter = iter(self.train_agent.train_loader)
            for step in range(1, self.train_batch_num + 1):
                total_step_num = (epoch - 1) * self.train_batch_num + step

                # data loading and preprocess part
                batch = next(train_iter)
                if not isinstance(batch, Dict):
                    raise TypeError(
                        f"collate_fn() of your Dataset should return a Dict, but got {type(batch)}!"
                    )
                batch = self.batch_preprocess(batch)

                # model forward and loss calculation part
                with autocast(enabled=self.use_amp):
                    train_losses = self.train_agent.train_step(batch, epoch=epoch, step=step)

                if not isinstance(train_losses, Dict):
                    raise TypeError(
                        "train_step() of your model should return a Dict of trainable torch.Tensor, "
                        f"but got {type(train_losses)}!"
                    )
                for key, value in train_losses.items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"The {key} loss from your train_step() should be a torch.Tensor, "
                            f"but got {type(value)}!"
                        )
                    if not value.requires_grad:
                        raise RuntimeError(f"The {key} loss from your train_step() is not trainable!")

                # loss backward and optimization
                # Hook: something to be done before the optimization step
                self.train_agent.before_optim_step()
                self.train_agent.optim_step(train_losses, total_step_num=total_step_num)
                # Hook: something to be done after the optimization step
                self.train_agent.after_optim_step()

                # make a copy to safely record training losses
                train_losses_detach = {key: value.clone().detach() for key, value in train_losses.items()}

                # gather all the losses from all DDP processes to rank0 and take average for recording
                if self.distributed:
                    for key in train_losses_detach.keys():
                        torch.distributed.reduce(
                            train_losses_detach[key], dst=0, op=torch.distributed.ReduceOp.SUM
                        )
                        train_losses_detach[key] /= torch.distributed.get_world_size()

                if train_pbar is not None:
                    train_pbar.update(1)
                    str_step_info = "Epoch: {epoch}/{epochs:4}. Step: {step}/{steps_per_epoch}. lr: {lr:.2e}".format(
                        epoch=epoch, epochs=self.train_epoch_num, step=step, steps_per_epoch=self.train_batch_num,
                        lr=self.train_agent.get_optim_lr()["main_opt"]
                    )
                    for l_name, l_value in train_losses_detach.items():
                        str_step_info += " {}: {:.4f}".format(l_name, l_value.item())
                    train_pbar.set_postfix_str(str_step_info)

            # skip validation if validation dataset is not specified by your TrainAgent
            if self.train_agent.valid_loader is not None:
                if isinstance(self.train_agent.valid_loader, Dict):
                    valid_iter = {
                        v_name: iter(v_loader) for v_name, v_loader in self.train_agent.valid_loader.items()
                    }
                else:
                    valid_iter = iter(self.train_agent.valid_loader)
                # validation on the main process (rank no.0) during training
                if is_main_proc():
                    # Hook: something to be done before each validation epoch
                    self.train_agent.before_valid_epoch()
                    self.train_agent.eval()
                    valid_results = self.eval(
                        eval_iterator=valid_iter,
                        total_eval_batch_num=self.valid_batch_num,
                        eval_step_fn=self.train_agent.valid_step,
                        eval_result_fn=self.train_agent.get_valid_results
                    )
                    # Hook: something to be done right after each validation epoch
                    self.train_agent.after_valid_epoch()

                    # update the best metric by the given selection criterion
                    if self.best_model_selection:
                        self.save_best_models(valid_results=valid_results, ckpt_prefix=f'epoch={epoch}')

                    # record the validation results to console
                    if isinstance(valid_iter, Dict):
                        valid_results = merge_flatten_dict(valid_results)
                    if is_main_proc():
                        print("Validation Results: ", dict_to_log_message(valid_results))

                    # activate the train mode right after validation
                    self.train_agent.train()

            # Hook: something to de done after each training epoch
            self.train_agent.after_train_epoch()

            # Synchronizes all rank processes here to avoid other ranks to move to the next train step early.
            if self.distributed:
                torch.distributed.barrier()

            if train_pbar is not None:
                train_pbar.clear()

        # clean up the allocated GPU memory after training for subsequent testing stage
        torch.cuda.empty_cache()

    def eval_single_iterator(
            self, iterator: Iterator, eval_step_fn: Callable, total_eval_batch_num: int, iterator_name: str = None
    ):
        with torch.no_grad():
            desc_string = f'{iterator_name} ' if iterator_name is not None else ''
            pbar = None
            if is_main_proc():
                pbar = tqdm(total=total_eval_batch_num, desc=f'{desc_string}eval', leave=False)

            for step in range(1, total_eval_batch_num + 1):
                batch = next(iterator)
                batch = self.batch_preprocess(batch)
                eval_step_fn(batch, iterator_name)
                if is_main_proc():
                    pbar.update(1)

            if pbar is not None:
                pbar.clear()

    def eval(
            self, eval_iterator: Union[Dict[str, Iterator], Iterator],
            total_eval_batch_num: Union[int, Dict[str, int]],
            eval_step_fn: Callable, eval_result_fn: Callable
    ):
        # a single dataset to be evaluated
        if isinstance(eval_iterator, Iterator):
            assert isinstance(total_eval_batch_num, int)
            self.eval_single_iterator(
                iterator=eval_iterator, eval_step_fn=eval_step_fn,
                total_eval_batch_num=total_eval_batch_num,
            )
            eval_results = eval_result_fn()

        # multiple datasets to be evaluated
        else:
            assert isinstance(total_eval_batch_num, Dict)
            for name, iterator in eval_iterator.items():
                self.eval_single_iterator(
                    iterator=iterator, iterator_name=name,
                    total_eval_batch_num=total_eval_batch_num[name],
                    eval_step_fn=eval_step_fn
                )

            eval_results = eval_result_fn()
            eval_results['average'] = average_dicts(eval_results)

        return eval_results

    def get_checkpoints_from_config(self) -> List[Union[str, None]]:
        if self.args.ckpt_path is not None:
            ckpt_path = self.args.ckpt_path
            if not ckpt_path.startswith('/'):
                ckpt_path = f"{self.launch_folder_path}/{self.args.ckpt_path}"
            if os.path.exists(ckpt_path):
                checkpoint_list = get_ckpt_by_path(ckpt_path)
            else:
                checkpoint_list = [None]
        else:
            checkpoint_list = [None]
        return checkpoint_list

    def test(self):

        def test_single_ckpt():
            if ckpt_path is not None:
                print(f'Start testing on the checkpoint:{ckpt_path}')
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.train_agent.load_model_state_dict(ckpt)
            else:
                print('No checkpoint is loaded into your model! Please be careful!')

            # Hook: something to be done before testing
            self.train_agent.before_test()
            self.train_agent.eval()
            if isinstance(self.train_agent.test_loader, Dict):
                test_iter = {
                    t_name: iter(t_loader) for t_name, t_loader in self.train_agent.test_loader.items()
                }
            else:
                test_iter = iter(self.train_agent.test_loader)

            test_results = self.eval(
                eval_iterator=test_iter,
                total_eval_batch_num=self.test_batch_num,
                eval_step_fn=self.train_agent.test_step,
                eval_result_fn=self.train_agent.get_test_results
            )
            # Hook: something to be done after testing
            self.train_agent.after_test()

            if isinstance(test_iter, Dict):
                test_results = merge_flatten_dict(test_results)
            if is_main_proc():
                print("Test Results: ", dict_to_log_message(test_results))

        checkpoint_list = self.get_checkpoints_from_config()
        for ckpt_path in checkpoint_list:
            test_single_ckpt()
