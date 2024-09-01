from typing import List
import torch


def fix_params(input_module: torch.nn.Module):
    for name, param in input_module.named_parameters():
        param.requires_grad = False


def data_instance_norm(data_list: List[torch.Tensor], norm_type: str):
    norm_data, norm_args = data_list, None
    # normalize the depth data by different norm method
    if norm_type == 'min-max':
        max_values = [torch.max(d.view(d.shape[0], -1), dim=-1, keepdim=True)[0].unsqueeze(-1) for d in data_list]
        min_values = [torch.min(d.view(d.shape[0], -1), dim=-1, keepdim=True)[0].unsqueeze(-1) for d in data_list]
        norm_data = [(d - min_values[i]) / (max_values[i] - min_values[i] + 1e-10) for i, d in enumerate(data_list)]
        norm_args = dict(min=min_values, max=max_values)

    elif norm_type == 'mean-std':
        mean_values = [torch.mean(d.view(d.shape[0], -1), dim=-1, keepdim=True).unsqueeze(-1) for d in data_list]
        std_values = [torch.std(d.view(d.shape[0], -1), dim=-1, keepdim=True).unsqueeze(-1) for d in data_list]
        norm_data = [(d - mean_values[i]) / (std_values[i] + 1e-10) for i, d in enumerate(data_list)]
        norm_args = dict(mean=mean_values, std=std_values)

    elif norm_type is not None:
        raise ValueError("Norm type must be either 'min-max' or 'mean-std'!")

    # do nothing if self.norm_type is None
    return norm_data, norm_args
