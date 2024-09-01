import json
import random
from os.path import join
from typing import Dict

import numpy as np
import torch
from PIL import Image

from mm_sam.datasets.base import BaseSAMDataset
from mm_sam.datasets import DATASETS
from mm_sam.datasets.gen_prompts import generate_prompts_from_object_masks

TRANSFER_DATASETS = DATASETS["CMTransfer"]
FUSION_DATASETS = DATASETS["MMFusion"]

from utilbox.demo_vis.vis_utils import nonrgb_to_rgb
from utilbox.global_config import DATA_ROOT
from utilbox.transforms import init_transforms_by_config
from utilbox.transforms.img_segm import Compose

city_split_dict = dict(
    # 2969 images
    train=['AddisAbaba', 'Barcelona', 'Brasilia', 'Jacksonville', 'NewDelhi', 'NewYork', 'SaoPaulo', 'Tokyo', 'Berlin', 'Darwin', 'Portsmouth', 'Rio', 'SanDiego', 'SaoLuis', 'Sydney'],
    # 751 images
    test=['Copenhagen', 'Suzhou']
)


class DFC23Dataset(BaseSAMDataset):
    semantic_classes = ["background", "building"]

    def __init__(
            self,
            is_train: bool,
            data_dir: str = f"{DATA_ROOT}/dfc23",
            use_corrected_mask: bool = False,
            transforms: Dict = None,
            image_type: str = 'rgb_images',
            **prompt_kwargs
    ) -> None:
        if image_type not in ['rgb_images', 'sar_images', 'sar_rgb_images']:
            raise ValueError(f"invalid image type: {image_type}")
        self.image_type = image_type

        metadata_path = join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        city_split = city_split_dict['train' if is_train else 'test']
        metadata = {key: value for key, value in metadata.items() if key.split('_')[1] in city_split}

        for key in metadata:
            metadata[key].update(
                rgb_file_path=data_dir + '/' + metadata[key]['rgb_file_path'],
                sar_file_path=data_dir + '/' + metadata[key]['sar_file_path'],
                gt_mask_path=data_dir + '/' + metadata[key]['gt_mask_path'],
                object_masks_path=data_dir + '/' + metadata[key]['object_masks_path']
            )
            if use_corrected_mask:
                metadata[key].update(
                    gt_mask_path=data_dir + '/' + metadata[key]['corr_gt_mask_path'],
                    object_masks_path=data_dir + '/' + metadata[key]['corr_object_masks_path']
                )
            metadata[key].pop('corr_gt_mask_path')
            metadata[key].pop('corr_object_masks_path')
            metadata[key]["image_path"] = metadata[key]["rgb_file_path"]

        super(DFC23Dataset, self).__init__(
            data_dict=metadata, is_train=is_train,  **prompt_kwargs
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

    def __getitem__(self, index: int):
        index_name = self.index_name_list[index]
        data_dict = self.data_dict[index_name]

        # (H, W, 3)
        rgb_image = Image.open(data_dict['rgb_file_path']).convert('RGB')
        rgb_image = np.array(rgb_image, dtype=np.float32)

        # (H, W,) -> (H, W, 1)
        sar_image = Image.open(data_dict['sar_file_path'])
        sar_image = np.array(sar_image, dtype=np.float32)[:, :, None]

        # (H, W)
        gt_mask = Image.open(data_dict['gt_mask_path']).convert('L')
        gt_mask = np.array(gt_mask, dtype=np.float32)
        gt_mask = np.where(gt_mask > 0., 1., 0.)

        # (N_obj, H, W)
        object_masks_path, data_field = data_dict['object_masks_path'].split('::')
        with np.load(object_masks_path) as npz_file:
            object_masks = npz_file[data_field]

        if self.transforms is not None:
            transformed = self.transforms(
                # (H, W) or (H, W, C) for uniform transform
                image=rgb_image, sar_image=sar_image, gt_mask=gt_mask, object_masks=object_masks.transpose(1, 2, 0)
            )
            rgb_image, sar_image, gt_mask, object_masks = (
                transformed["image"], transformed['sar_image'], transformed['gt_mask'], transformed['object_masks']
            )
            # (H, W, N_obj) -> (N_obj, H, W)
            object_masks = object_masks.transpose(2, 0, 1)

        if self.image_type == 'rgb_images':
            image_shape = (rgb_image.shape[0], rgb_image.shape[1])
        elif self.image_type in ['sar_images', 'sar_rgb_images']:
            image_shape = (sar_image.shape[0], sar_image.shape[1])
        else:
            raise RuntimeError(f"Meet wrong image type during training: {self.image_type}!")
        point_coords, box_coords, noisy_object_masks, object_masks = generate_prompts_from_object_masks(
            image_shape=image_shape, object_masks=object_masks,
            tgt_prompts=[random.choice(self.used_prompts)], **self.prompt_kwargs
        )

        # (H, W, 3)
        sar_rgb_image = nonrgb_to_rgb(sar_image)
        ret_dict = dict(
            rgb_images=torch.from_numpy(rgb_image).to(dtype=torch.float32).permute(2, 0, 1),
            sar_images=torch.from_numpy(sar_image).to(dtype=torch.float32).permute(2, 0, 1),
            sar_rgb_images=torch.from_numpy(sar_rgb_image).to(dtype=torch.float32).permute(2, 0, 1),
            gt_masks=torch.from_numpy(gt_mask),
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks,
            index_name=index_name
        )
        ret_dict['images'] = ret_dict[self.image_type]
        return ret_dict

@TRANSFER_DATASETS.register("dfc23")
class DFC23Transfer(DFC23Dataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="sar_images", **init_kwargs)


@FUSION_DATASETS.register("dfc23")
class DFC23Fusion(DFC23Dataset):
    def __init__(self, is_train: bool, max_object_num: int = 50, **init_kwargs):
        if not is_train:
            max_object_num = None
        super().__init__(is_train=is_train, image_type="rgb_images", max_object_num=max_object_num, **init_kwargs)
