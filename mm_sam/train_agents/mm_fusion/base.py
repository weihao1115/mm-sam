import os.path
from os.path import join
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from mm_sam.datasets import DATASETS
from mm_sam.datasets.transforms import TRANSFORMS_DICT
from mm_sam.models.sam import SAMWrapper
from mm_sam.train_agents.cm_transfer.base import XLoraEncoder
from mm_sam.models.module_lib.sfg import SelectiveFusionGate
from mm_sam.train_agents.sam import BaseSAMTrainAgent
from utilbox.global_config import EXP_ROOT
from utilbox.optim_agents.standard import StandardOptimAgent
from utilbox.schedulers.torch_wrappers import CosineAnnealingLR
from utilbox.train_managers.base import get_ckpt_by_path
from utilbox.train_utils import fix_params


@torch.no_grad()
def infer_by_max_confid(mask_logits_list: List[torch.Tensor]):
    # confidence is measured by the distance from pre-sigmoid logits to the origin
    mask_confids = torch.stack([torch.abs(m_logits) for m_logits in mask_logits_list], dim=0)
    argmax_confid = torch.argmax(mask_confids, dim=0)

    max_confid_mask = torch.zeros_like(argmax_confid, dtype=torch.float32)
    for i, mask_logits in enumerate(mask_logits_list):
        max_confid_flags = argmax_confid == i
        max_confid_mask[max_confid_flags] = mask_logits[max_confid_flags]
    return max_confid_mask


def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def compute_single_mask_loss(pred: torch.Tensor, label: torch.Tensor):
    label = torch.where(torch.gt(label, 0.), 1., 0.)
    b_loss = F.binary_cross_entropy_with_logits(pred, label.float())
    d_loss = calculate_dice_loss(pred, label)
    return b_loss, d_loss


def compute_sam_sup_loss(
        masks_pred: Union[torch.Tensor, List[torch.Tensor]],
        masks_gt: Union[torch.Tensor, List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # check of mask shapes before loss calculation
    for mask, label in zip(masks_pred, masks_gt):
        if mask.shape[-2:] != label.shape[-2:]:
            raise RuntimeError(
                "Postprocessed predicted masks and ground-truth labels have different shapes!"
                f"Got masks are {mask.shape[-2:]} while labels are {label.shape[-2:]}!"
            )
    for masks in [masks_pred, masks_gt]:
        for i in range(len(masks)):
            if len(masks[i].shape) == 2:
                masks[i] = masks[i][None, None, :]
            if len(masks[i].shape) == 3:
                masks[i] = masks[i][:, None, :]
            if len(masks[i].shape) != 4:
                raise RuntimeError

    if isinstance(masks_pred, torch.Tensor):
        bce_loss, dice_loss = compute_single_mask_loss(masks_pred, masks_gt)
    elif isinstance(masks_pred, List):
        bce_loss_list, dice_loss_list = [], []
        for i in range(len(masks_pred)):
            _bce_loss, _dice_loss = compute_single_mask_loss(masks_pred[i], masks_gt[i])
            bce_loss_list.append(_bce_loss)
            dice_loss_list.append(_dice_loss)
        bce_loss = sum(bce_loss_list) / len(bce_loss_list)
        dice_loss = sum(dice_loss_list) / len(dice_loss_list)
    else:
        raise RuntimeError

    return bce_loss, dice_loss


class MMFusionSAM(BaseSAMTrainAgent):

    def agent_init(
            self,
            train_data: str,
            # align encoder-related arguments
            x_data_field: str,
            x_channel_num: int,
            x_encoder_ckpt_path: str,
            x_lora_rank: int = 4,
            x_norm_type: str = 'mean-std',
            # SFG arguments
            sfg_filter_num: int = 1,
            sfg_inter_channels: int = 32,
            sfg_filter_type: str = "conv2d",
            # SAM arguments
            sam_model_type: str = "vit_b",
            # data arguments
            train_transforms: Optional[str] = None,
            valid_transforms: Optional[str] = None,
            test_transforms: Optional[str] = None,
            valid_data: Optional[str] = None,
            test_data: Optional[str] = None,
            **kwargs
    ):
        # build datasets
        if valid_data is None: valid_data = train_data
        if test_data is None: test_data = valid_data
        self.build_datasets(train_data, train_transforms, valid_data, valid_transforms, test_data, test_transforms)

        # build SAM backbone
        self.sam = SAMWrapper(model_type=sam_model_type)

        # align encoder init & params load
        self.x_encoder = XLoraEncoder(
            x_data_field=x_data_field, x_channel_num=x_channel_num, lora_rank=x_lora_rank,
            norm_type=x_norm_type, rgb_encoder=self.sam.image_encoder
        )
        if not x_encoder_ckpt_path.startswith("/"):
            x_encoder_ckpt_path = join(EXP_ROOT, x_encoder_ckpt_path)
        if os.path.isdir(x_encoder_ckpt_path):
            ckpt_list = get_ckpt_by_path(x_encoder_ckpt_path)
            if len(ckpt_list) > 1:
                raise RuntimeError(
                    f"Multiple checkpoints are found in {x_encoder_ckpt_path}! "
                    "Please directly specify the path of the checkpoint file you target!"
                )
            elif len(ckpt_list) == 0:
                raise RuntimeError(f"No checkpoint found in {x_encoder_ckpt_path}!")
            x_encoder_ckpt_path = ckpt_list[0]
        ckpt = torch.load(x_encoder_ckpt_path, map_location='cpu')
        self.x_encoder.load_state_dict(ckpt)
        fix_params(self.x_encoder)

        self.fusion_module = SelectiveFusionGate(
            filter_num=sfg_filter_num, intermediate_channels=sfg_inter_channels, filter_type=sfg_filter_type
        )

        optimizer = torch.optim.AdamW(params=self.fusion_module.parameters(), lr=4e-4)
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, batches_per_epoch=self.train_batch_num,
            max_epochs=self.manager_config['train_epoch_num'], min_lr=1e-5
        )
        self.optim_agent = StandardOptimAgent(
            optimizer=optimizer, scheduler=scheduler, use_amp=self.manager_config['use_amp']
        )

    def build_datasets(
            self,
            train_data: str, train_transforms: str,
            valid_data: str, valid_transforms: str,
            test_data: str, test_transforms: str,
    ):
        if train_transforms is not None:
            train_transforms = TRANSFORMS_DICT[train_transforms]
        self.train_dataset = DATASETS["MMFusion"].build(
            train_data, kwargs=dict(transforms=train_transforms, is_train=True)
        )
        if valid_transforms is not None:
            valid_transforms = TRANSFORMS_DICT[valid_transforms]
        self.valid_dataset = DATASETS["MMFusion"].build(
            valid_data, kwargs=dict(transforms=valid_transforms, is_train=False)
        )
        if test_transforms is not None:
            test_transforms = TRANSFORMS_DICT[test_transforms]
        self.test_dataset = DATASETS["MMFusion"].build(
            test_data, kwargs=dict(transforms=test_transforms, is_train=False)
        )

    def encode(self, data_dict: Dict):
        # Process RGB-modality Data (non-trainable)
        with torch.no_grad():
            # here we register the ori_infer_img_size in self.sam by rgb_images (for subsequent self.sam.infer calling)
            if hasattr(self.sam, "module"): set_infer_img_fn = self.sam.module.set_infer_img
            else: set_infer_img_fn = self.sam.set_infer_img
            set_infer_img_fn(img=data_dict['rgb_images'])
            rgb_feats = self.sam.img_features
            rgb_feats = rgb_feats.detach()

        # Process X-modality Data
        # x_images in original shapes after pre-normalization (if specified)
        if hasattr(self.x_encoder, "module"): x_preprocess_fn = self.x_encoder.module.preprocess
        else: x_preprocess_fn = self.x_encoder.preprocess
        x_images = x_preprocess_fn(data_dict=data_dict)
        # x_feats: (B, 256, 64, 64)
        x_feats = self.x_encoder(x_images)

        # fusion the RGB and X features as the encoder embedding for mask prediction
        fuse_feats, fuse_feat_masks = self.fusion_module(feat_list=[rgb_feats, x_feats])
        return rgb_feats, x_feats, fuse_feats, fuse_feat_masks

    def set_infer_img(self, img=None, data_dict: Dict = None):
        assert data_dict is not None, f'data_dict must be given in {self.__class__.__name__}!'
        _, _, fuse_feats, _ = self.encode(data_dict=data_dict)
        self.sam.img_features = fuse_feats

    def train_step(self, batch: Dict, epoch: int, step: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Training will be released soon~")

    def model_state_dict(self) -> Dict:
        model = self.fusion_module
        if hasattr(model, 'module'):
            model = model.module
        return model.state_dict()

    def load_model_state_dict(self, model_state_dict: Dict):
        model = self.fusion_module
        if hasattr(model, 'module'):
            model = model.module
        model.load_state_dict(model_state_dict)
