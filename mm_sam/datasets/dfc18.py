import json
from os.path import join
from typing import Dict, Optional

import numpy as np
import torch

from mm_sam.datasets.base import BaseSAMDataset
from mm_sam.datasets import DATASETS
TRANSFER_DATASETS = DATASETS["CMTransfer"]
FUSION_DATASETS = DATASETS["MMFusion"]

from utilbox.demo_vis.vis_utils import nonrgb_to_rgb
from utilbox.global_config import DATA_ROOT
from utilbox.transforms import init_transforms_by_config
from utilbox.transforms.img_segm import Compose


class DFC18Dataset(BaseSAMDataset):
    semantic_classes = ["background", "building"]

    def __init__(
            self,
            is_train: bool,
            data_dir: str = f"{DATA_ROOT}/dfc18_dump",
            image_type: str = 'rgb_images',
            transforms: Optional[Dict] = None,
            area_threshold: int = 100,
            max_side_ratio: Optional[float] = 6.,
            **prompt_kwargs
    ):
        if image_type not in ['rgb_images', 'hsi_images', 'hsi_rgb_images', 'proj_xyz3c_images', 'proj_3c_rgb_images']:
            raise ValueError(f"invalid image type: {image_type}")
        self.image_type = image_type

        json_path = join(data_dir, 'train.json' if is_train else 'test.json')
        with open(json_path, 'r') as j_f:
            data_dict = json.load(j_f)

        # recover relative path to absolute ones
        for key in data_dict.keys():
            data_dict[key]['npz_path'] = join(data_dir, data_dict[key]['npz_path'])
        super(DFC18Dataset, self).__init__(
            data_dict=data_dict, is_train=is_train, area_threshold=area_threshold, max_side_ratio=max_side_ratio,
            **prompt_kwargs
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
        index_name = self.index_name_list[index]
        npz_path = self.data_dict[index_name]['npz_path']
        # Note: use context to properly release the handle of npz files
        with np.load(npz_path) as index_npz_file:
            rgb_image, hsi_image, proj_xyz3c_image, gt_mask = (
                index_npz_file['rgb_image'], index_npz_file['hsi_image'],
                index_npz_file['proj_xyz3c_image'], index_npz_file['proj_sem_mask' if self.is_train else 'gt_mask']
            )

        if self.is_train:
            # turn semantic PC projection mask into binary mask for buildings (no.6 & 17 classes)
            gt_mask = np.where(np.logical_or(gt_mask == 6, gt_mask == 17), 1., 0.)
        else:
            # turn semantic GT mask into binary mask for buildings (no.8 & 9 classes)
            gt_mask = np.where(np.logical_or(gt_mask == 8, gt_mask == 9), 1., 0.)

        if self.transforms is not None:
            transformed = self.transforms(
                image=rgb_image, hsi_image=hsi_image, proj_xyz3c_image=proj_xyz3c_image, gt_mask=gt_mask
            )
            rgb_image, hsi_image, proj_xyz3c_image, gt_mask = (
                transformed["image"], transformed['hsi_image'],
                transformed['proj_xyz3c_image'], transformed['gt_mask']
            )

        if self.image_type == 'rgb_images':
            image_shape = (rgb_image.shape[0], rgb_image.shape[1])
        elif self.image_type in ['hsi_images', 'hsi_rgb_images']:
            image_shape = (hsi_image.shape[0], hsi_image.shape[1])
        elif self.image_type in ['proj_xyz3c_images', 'proj_3c_rgb_images']:
            image_shape = (proj_xyz3c_image.shape[0], proj_xyz3c_image.shape[1])
        else:
            raise RuntimeError(f"Meet wrong image type during training: {self.image_type}!")
        point_coords, box_coords, noisy_object_masks, object_masks, _ = (
            self.get_prompts_from_gt_masks(gt_mask=gt_mask, image_shape=image_shape)
        )

        # the last two channels are reserved for X and Y info
        hsi_image = hsi_image[:, :, :-2]
        hsi_rgb_image = nonrgb_to_rgb(hsi_image[:, :, [15, 31, 47]])
        proj_3c_rgb_image = nonrgb_to_rgb(proj_xyz3c_image[:, :, 3:])
        ret_dict = dict(
            rgb_images=torch.from_numpy(rgb_image).to(dtype=torch.float32).permute(2, 0, 1),
            hsi_images=torch.from_numpy(hsi_image).to(dtype=torch.float32).permute(2, 0, 1),
            hsi_rgb_images=torch.from_numpy(hsi_rgb_image).to(dtype=torch.float32).permute(2, 0, 1),
            proj_xyz3c_images=torch.from_numpy(proj_xyz3c_image).to(dtype=torch.float32).permute(2, 0, 1),
            proj_3c_rgb_images=torch.from_numpy(proj_3c_rgb_image).to(dtype=torch.float32).permute(2, 0, 1),
            gt_masks=torch.from_numpy(gt_mask),
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks,
            index_name=index_name
        )
        ret_dict['images'] = ret_dict[self.image_type]
        return ret_dict


@TRANSFER_DATASETS.register("dfc18_hsi")
class DFC18HSITransfer(DFC18Dataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="hsi_images", **init_kwargs)


@TRANSFER_DATASETS.register("dfc18_pc")
class DFC18PCTransfer(DFC18Dataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="proj_xyz3c_images", **init_kwargs)


@FUSION_DATASETS.register("dfc18")
class DFC18Fusion(DFC18Dataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="rgb_images", **init_kwargs)
