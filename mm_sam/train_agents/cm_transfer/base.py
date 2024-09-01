from typing import Dict, Union, List, Optional

import torch.nn.functional as F
import torch

from mm_sam.datasets import DATASETS
from mm_sam.datasets.transforms import TRANSFORMS_DICT
from mm_sam.models.module_lib.x_encoder import XLoraEncoder
from mm_sam.models.sam import SAMWrapper
from mm_sam.train_agents.sam import BaseSAMTrainAgent

from utilbox.optim_agents.standard import StandardOptimAgent
from utilbox.schedulers.torch_wrappers import CosineAnnealingLR
from utilbox.train_utils import fix_params


class CMTransferSAM(BaseSAMTrainAgent):
    def agent_init(
            self,
            train_data: str,
            # x encoder arguments
            x_data_field: str,
            x_channel_num: int,
            x_lora_rank: int = 4,
            x_norm_type: Union[str, None] = 'mean-std',
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

        # backup the original image encoder for the RGB branch
        assert hasattr(self.sam, 'image_encoder'), "image_encoder has not been initialized in your given sam_model!"
        self.rgb_encoder = self.sam.image_encoder
        fix_params(self.rgb_encoder)  # frozen the RGB branch

        # replace the position of image encoder by the X-align encoder for the X branch
        self.x_encoder = XLoraEncoder(
            x_data_field=x_data_field, x_channel_num=x_channel_num, lora_rank=x_lora_rank,
            norm_type=x_norm_type, rgb_encoder=self.rgb_encoder
        )
        self.sam.image_encoder = self.x_encoder
        self.x_data_field = x_data_field

        optimizer = torch.optim.AdamW(params=self.x_encoder.parameters(), lr=1.6e-3)
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
        self.train_dataset = DATASETS["CMTransfer"].build(
            train_data, kwargs=dict(transforms=train_transforms, is_train=True)
        )
        if valid_transforms is not None:
            valid_transforms = TRANSFORMS_DICT[valid_transforms]
        self.valid_dataset = DATASETS["CMTransfer"].build(
            valid_data, kwargs=dict(transforms=valid_transforms, is_train=False)
        )
        if test_transforms is not None:
            test_transforms = TRANSFORMS_DICT[test_transforms]
        self.test_dataset = DATASETS["CMTransfer"].build(
            test_data, kwargs=dict(transforms=test_transforms, is_train=False)
        )

    def preprocess_x_images(self, x_images: List[torch.Tensor]) -> List[torch.Tensor]:
        return x_images

    def set_infer_img(self, img=None, data_dict: Dict = None):
        data_dict[self.x_data_field] = self.preprocess_x_images(data_dict[self.x_data_field])

        if hasattr(self.x_encoder, "module"): x_preprocess_fn = self.x_encoder.module.preprocess
        else: x_preprocess_fn = self.x_encoder.preprocess
        x_images = x_preprocess_fn(data_dict=data_dict)

        if hasattr(self.sam, "module"): set_infer_img_fn = self.sam.module.set_infer_img
        else: set_infer_img_fn = self.sam.set_infer_img
        set_infer_img_fn(img=x_images, pixel_norm=False)  # SAM RGB pre-norm is disabled

    def train_step(self, batch: Dict, epoch: int, step: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Training will be released soon~")

    def model_state_dict(self) -> Dict:
        model = self.x_encoder
        if hasattr(model, 'module'):
            model = model.module
        return model.state_dict()

    def load_model_state_dict(self, model_state_dict: Dict):
        model = self.x_encoder
        if hasattr(model, 'module'):
            model = model.module
        model.load_state_dict(model_state_dict)
