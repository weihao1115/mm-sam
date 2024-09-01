from typing import Dict, Union, Callable, Optional
import torch
from mm_sam.datasets.base import BaseSAMDataset
from mm_sam.models.sam import SAMWrapper

from utilbox.evaluators.iou import StreamSegMetrics
from utilbox.train_agents.base import TrainAgent


class BaseSAMTrainAgent(TrainAgent):
    """Base class for Sam-based Models."""
    def after_agent_init(self, evaluator_cls: Callable = StreamSegMetrics, **kwargs):
        def build_evaluator_from_dataset(eva_dataset: Union[BaseSAMDataset, Dict[str, BaseSAMDataset]]):
            if isinstance(eva_dataset, Dict):
                evaluator = {
                    e_key: evaluator_cls(class_names=e_data.semantic_classes)
                    for e_key, e_data in eva_dataset.items()
                }
            else:
                evaluator = evaluator_cls(class_names=eva_dataset.semantic_classes)
            return evaluator
        self.valid_evaluator = build_evaluator_from_dataset(self.valid_dataset)
        self.test_evaluator = build_evaluator_from_dataset(self.test_dataset)

    @property
    def sam(self) -> SAMWrapper:
        if hasattr(self, '_sam'):
            return self._sam
        else:
            raise RuntimeError("Please register the SAM model by `self.sam = ...` in your agent_init()!")

    @sam.setter
    def sam(self, model: SAMWrapper):
        assert isinstance(model, SAMWrapper), "Your registered SAM model must be an instance of SAMWrapper!"
        self._sam = model

    @property
    def mask_threshold(self) -> float:
        if self.distributed:
            return self.sam.module.mask_threshold
        else:
            return self.sam.mask_threshold

    def train_step(self, batch: Dict, epoch: int, step: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def valid_step(self, batch: Dict, iter_name: Optional[str] = None, evaluator: Optional[StreamSegMetrics] = None):
        box_coords, gt_masks, index_name = batch['box_coords'], batch['gt_masks'], batch['index_name']
        self.set_infer_img(data_dict=batch)

        # binary segmentation
        if 'object_classes' not in batch:
            box_masks_pred, _ = self.infer(
                box_coords=box_coords,
                output_mask_size=[(mask.shape[-2], mask.shape[-1]) for mask in batch['gt_masks']]
            )
            # Add some logics here to deal with the case that no box is given #
            for i in range(len(box_masks_pred)):
                if box_coords[i] is None:
                    # set the prediction to zeros if no box is given (i.e., no foreground prediction)
                    box_masks_pred[i] = torch.zeros_like(gt_masks[i])

        # semantic segmentation
        else:
            object_classes = batch['object_classes']
            box_masks_pred, _ = self.infer(
                box_coords=box_coords,
                output_mask_size=[(mask.shape[-2], mask.shape[-1]) for mask in batch['gt_masks']],
                return_all_prompt_masks=True
            )
            for i in range(len(box_masks_pred)):
                if box_coords[i] is not None:
                    for j in range(len(box_masks_pred[i])):
                        box_masks_pred[i][j] *= object_classes[i][j]
                    # set the duplicated predictions to the largest semantic class idx
                    box_masks_pred[i] = torch.max(box_masks_pred[i], dim=0, keepdim=True)[0]
                else:
                    # set the prediction to zeros if no box is given (i.e., no foreground prediction)
                    box_masks_pred[i] = torch.zeros_like(gt_masks[i])

        if evaluator is None:
            evaluator = self.valid_evaluator
        if iter_name is not None:
            evaluator = evaluator[iter_name]
        evaluator.update(label_trues=gt_masks, label_preds=box_masks_pred, index_name=index_name)

    def get_valid_results(self, evaluator = None) -> Dict:
        if evaluator is None:
            evaluator = self.valid_evaluator

        valid_results = {}
        if isinstance(evaluator, Dict):
            for v_key, v_eva in evaluator.items():
                valid_results[v_key] = v_eva.compute()[0]
                v_eva.reset()
        else:
            valid_results = evaluator.compute()[0]
            evaluator.reset()
        return valid_results

    def test_step(self, batch: Dict, iter_name: str = None):
        self.valid_step(batch, iter_name, self.test_evaluator)

    def get_test_results(self) -> (Dict, Dict):
        return self.get_valid_results(self.test_evaluator)

    def set_infer_img(self, *args, **kwargs):
        if hasattr(self.sam, "module"):
            return self.sam.module.set_infer_img(*args, **kwargs)
        else:
            return self.sam.set_infer_img(*args, **kwargs)

    def infer(self, *args, **kwargs):
        if hasattr(self.sam, "module"):
            return self.sam.module.infer(*args, **kwargs)
        else:
            return self.sam.infer(*args, **kwargs)

    @property
    def ori_infer_img_size(self):
        if hasattr(self.sam, "module"):
            return self.sam.module.ori_infer_img_size
        else:
            return self.sam.ori_infer_img_size

    @property
    def img_features(self):
        if hasattr(self.sam, "module"):
            return self.sam.module.img_features
        else:
            return self.sam.img_features
