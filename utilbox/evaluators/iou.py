from typing import List, Dict, Tuple, Any, Union

import numpy as np
import torch

from utilbox.evaluators.misc import label_pred_true_preprocess
from utilbox.evaluators.base import BaseEvaluator


class StreamSegMetrics(BaseEvaluator):
    """Stream Metrics for Semantic Segmentation Task"""
    def __init__(self, class_names: List[str], chunk_size: int = None):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.chunk_size = chunk_size
        self.reset()

    def update(
            self,
            label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            index_name: Union[List[Any], Any]
    ):
        label_preds, label_trues = label_pred_true_preprocess(label_preds, label_trues)
        if not isinstance(index_name, List):
            index_name = [index_name]
        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            lt, lp = lt.flatten(), lp.flatten()
            if self.chunk_size is None:
                lt, lp = [lt], [lp]
            else:
                chunk_num = len(lt) // self.chunk_size + 1
                lt = [lt[i * self.chunk_size: min((i + 1) * self.chunk_size, len(lt))] for i in range(chunk_num)]
                lp = [lp[i * self.chunk_size: min((i + 1) * self.chunk_size, len(lp))] for i in range(chunk_num)]

            ori_confusion_matrix = self.confusion_matrix.copy()
            for l_t, l_p in zip(lt, lp):
                index_hist = self._fast_hist(l_t, l_p)
                self.confusion_matrix += index_hist
            index_confusion_matrix = self.confusion_matrix - ori_confusion_matrix
            self.index_results[index_name[i]] = self.compute(hist=index_confusion_matrix)[0]

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def compute(self, hist=None) -> Tuple[Dict, Dict] or Tuple[None, None]:
        if hist is None: hist = self.confusion_matrix
        if hist.sum() == 0.: return None, None

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        fore_iu = iu[1:]
        mean_fore_iu = np.nanmean(fore_iu)
        nonzero_fore_iu = fore_iu[fore_iu > 0.0]
        mean_nonzero_fore_iu = np.nanmean(nonzero_fore_iu) if len(nonzero_fore_iu) > 0 else 0.0

        results_dict = {
            "mean_iou": mean_iu,
            "mean_fore_iu": mean_fore_iu,
            "mean_nonzero_fore_iu": mean_nonzero_fore_iu
        }
        for i, class_name in enumerate(self.class_names):
            results_dict[f'{class_name}_iou'] = iu[i]
        return results_dict, self.index_results

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.index_results = {}


class PointCloudSegMetrics(BaseEvaluator):
    """
    IoU Metric Evaluator specific for Point Cloud Segmentation
    """
    def __init__(self, class_names, ignore_label=255, skip_background: bool = False):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.ignore_label = ignore_label
        self.skip_background = skip_background
        self.reset()

    def calculate_eval_samples(self, labels_pred, labels_true):
        """
        Calculate evaluation samples for segmentation.

        Args:
            labels_pred (np.ndarray): Predicted labels.
            labels_true (np.ndarray): Ground truth labels.

        Returns:
            tuple: A tuple containing IoUs, mIoU, total_seen, total_correct, and total_positive.
        """
        labels_pred = labels_pred[labels_true != self.ignore_label]
        labels_true = labels_true[labels_true != self.ignore_label]

        total_seen = np.zeros(self.n_classes, int)
        total_correct = np.zeros(self.n_classes, int)
        total_positive = np.zeros(self.n_classes, int)

        for i in range(self.n_classes):
            total_seen[i] += np.sum(labels_true == i).item()
            total_correct[i] += np.sum((labels_true == i) & (labels_pred == labels_true)).item()
            total_positive[i] += np.sum(labels_pred == i).item()

        return total_seen, total_correct, total_positive

    def calculate_iou(self, total_seen, total_correct, total_positive):
        """
        Calculate IoUs and mIoU for segmentation.

        Args:
            total_seen (np.ndarray): Total number of ground truth labels per class.
            total_correct (np.ndarray): Total number of correctly predicted labels per class.
            total_positive (np.ndarray): Total number of predicted labels per class.

        Returns:
            tuple: A tuple containing IoUs and mIoU.
        """
        ious = [
            total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i]) if total_seen[i] != 0 else 1
            for i in range(self.n_classes)
        ]
        for i in range(len(ious)):
            if ious[i] == 1:
                ious[i] = 0
        miou = np.mean(ious[1:] if self.skip_background else ious)
        return ious, miou

    def update(
            self,
            label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            index_name: List
    ):
        label_preds, label_trues = label_pred_true_preprocess(label_preds, label_trues)
        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            res_seen, res_correct, res_positive = self.calculate_eval_samples(lp, lt)
            self.total_seen += res_seen
            self.total_correct += res_correct
            self.total_positive += res_positive

            index_ious, index_miou = self.calculate_iou(res_seen, res_correct, res_positive)
            self.index_results[index_name[i]] = dict(miou=index_miou)
            for class_name, class_iou in zip(self.class_names, index_ious):
                self.index_results[index_name[i]][f'{class_name}_iou'] = class_iou

    def compute(self) -> (Dict, Dict):
        ious, miou = self.calculate_iou(self.total_seen, self.total_correct, self.total_positive)
        results_dict = dict(miou=miou)
        for class_name, class_iou in zip(self.class_names, ious):
            results_dict[class_name] = class_iou
        return results_dict, self.index_results

    def reset(self, *args, **kwargs):
        self.index_results = {}
        self.total_seen = np.zeros(self.n_classes, int)
        self.total_correct = np.zeros(self.n_classes, int)
        self.total_positive = np.zeros(self.n_classes, int)
