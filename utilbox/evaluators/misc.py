import copy
from typing import List, Union

import numpy as np
import torch

from utilbox.type_utils import to_numpy


def label_pred_true_preprocess(
        label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
        label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]]
) -> (List[np.ndarray], List[np.ndarray]):

    def flatten_labels(labels):
        labels = copy.deepcopy(labels) # out-place operation
        if isinstance(labels, torch.Tensor):
            labels = to_numpy(labels)
        if not isinstance(labels, list):
            if len(labels.shape) in [1, 2]:
                labels = [labels]
            elif len(labels.shape) == 3:
                labels = [i for i in labels]
            else:
                raise RuntimeError("Unexpected label shape")

        for i in range(len(labels)):
            # from torch.Tensor to np.ndarray
            if not isinstance(labels[i], np.ndarray):
                labels[i] = to_numpy(labels[i])
            # ensure the mask is a 1d vector with int-type
            labels[i] = labels[i].reshape(-1).astype(int)
        return labels

    label_preds = flatten_labels(label_preds)
    label_trues = flatten_labels(label_trues)
    for pred, label in zip(label_preds, label_trues):
        if len(pred) != len(label):
            raise RuntimeError(
                "Predicted masks and ground-truth labels have different lengths!"
                f"Got masks are {len(pred)} while labels are {len(label)}!"
            )
    return label_preds, label_trues
