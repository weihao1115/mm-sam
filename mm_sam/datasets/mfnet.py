from os.path import join
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from mm_sam.datasets import DATASETS
TRANSFER_DATASETS = DATASETS["CMTransfer"]
FUSION_DATASETS = DATASETS["MMFusion"]

from mm_sam.datasets.base import BaseSAMDataset
from utilbox.demo_vis.vis_utils import nonrgb_to_rgb
from utilbox.global_config import DATA_ROOT
from utilbox.transforms import init_transforms_by_config
from utilbox.transforms.img_segm import Compose



class MFNetDataset(BaseSAMDataset):
    semantic_classes = [
        'unlabeled',
        'car',
        'person',
        'bike',
        'curve',
        'car_stop',
        'guardrail',
        'color_cone',
        'bump',
    ]

    def __init__(
            self,
            is_train: bool,
            data_dir: str = f"{DATA_ROOT}/MFNet",
            subset: Optional[str] = None,
            image_type: str = 'rgb_images',
            transforms: Union[Dict, Compose, None] = None,
            **prompt_kwargs
    ):
        if image_type not in ['rgb_images', 'thermal_images', 'thermal_rgb_images']:
            raise ValueError("Invalid image_type! Must be one of 'rgb_images', 'thermal_images', 'thermal_rgb_images'.")
        self.image_type = image_type

        if is_train:
            assert subset is None, "train_agents does not support day-night separation!"
            data_index_file = "train.txt"
        elif subset is None:
            data_index_file = "test.txt"
        else:
            assert subset in ["day", "night"], "Invalid subset! Must be either 'day' or 'night'"
            data_index_file = f"test_{subset}.txt"

        with open(join(data_dir, data_index_file), 'r') as f:
            data_index_list = [data_name.replace('\n', '') for data_name in f.readlines()]
            data_dict = {
                index_name: {
                    "image_path": join(data_dir, 'images', f'{index_name}.png'),
                    "gt_mask_path": join(data_dir, 'labels', f'{index_name}.png')
                } for index_name in data_index_list
            }
        super(MFNetDataset, self).__init__(
            data_dict=data_dict, is_train=is_train, label_threshold=None, **prompt_kwargs
        )

        # register the aligned transforms for RGB-Thermal-MASK triples
        if isinstance(transforms, Dict):
            self.transforms = init_transforms_by_config(
                transform_config=transforms, tgt_package="utilbox.transforms.img_segm",
                default_args=self.transforms_default_args
            )
        elif isinstance(transforms, Compose):
            self.transforms = transforms
        else:
            self.transforms = None

    def __getitem__(self, index):
        ret_dict = super().__getitem__(index)
        rgb_image = ret_dict['images'][:3].permute(1, 2, 0).numpy()
        thermal_image = ret_dict['images'][3].unsqueeze(0).permute(1, 2, 0).numpy()
        gt_mask = ret_dict['gt_masks'].numpy()

        if self.transforms is not None:
            transformed = self.transforms(image=rgb_image, thermal=thermal_image, mask=gt_mask)
            rgb_image, thermal_image, gt_mask = transformed["image"], transformed['thermal'], transformed['mask']

        if self.image_type == 'rgb_images':
            image_shape = (rgb_image.shape[0], rgb_image.shape[1])
        elif self.image_type in ['thermal_images', 'thermal_rgb_images']:
            image_shape = (thermal_image.shape[0], thermal_image.shape[1])
        else:
            raise RuntimeError(f"Meet wrong image type during training: {self.image_type}!")
        point_coords, box_coords, noisy_object_masks, object_masks, object_classes = (
            self.get_prompts_from_gt_masks(gt_mask=gt_mask, image_shape=image_shape)
        )

        thermal_rgb_image = nonrgb_to_rgb(thermal_image)
        ret_dict.update(
            rgb_images=torch.from_numpy(rgb_image).to(torch.float32).permute(2, 0, 1),
            thermal_images=torch.from_numpy(thermal_image).to(torch.float32).permute(2, 0, 1),
            thermal_rgb_images=torch.from_numpy(thermal_rgb_image).to(torch.float32).permute(2, 0, 1),
            gt_masks=torch.from_numpy(gt_mask),
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks,
            object_classes=object_classes
        )
        ret_dict['images'] = ret_dict[self.image_type]
        return ret_dict

    def get_image_by_path(self, image_path: str):
        rgb_t = Image.open(image_path)
        rgb_t = np.array(rgb_t, dtype=np.float32)
        return rgb_t


@TRANSFER_DATASETS.register("mfnet")
class MFNetTransfer(MFNetDataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="thermal_images", **init_kwargs)


@FUSION_DATASETS.register("mfnet")
class MFNetFusion(MFNetDataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="rgb_images", **init_kwargs)
