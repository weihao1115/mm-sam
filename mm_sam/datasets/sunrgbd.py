from os.path import join
from typing import Dict, Union

import torch

from mm_sam.datasets import DATASETS
TRANSFER_DATASETS = DATASETS["CMTransfer"]
FUSION_DATASETS = DATASETS["MMFusion"]

from mm_sam.datasets.base import BaseSAMDataset
from utilbox.data_load.read_utils import read_depth_from_disk
from utilbox.demo_vis.vis_utils import nonrgb_to_rgb
from utilbox.global_config import DATA_ROOT
from utilbox.transforms import init_transforms_by_config
from utilbox.transforms.img_segm import Compose


def get_idx2data_dict(data_dir: str, train_flag: bool) -> (Dict, Dict, Dict):
    depth_path_file = 'train_depth.txt' if train_flag else 'test_depth.txt'
    with open(join(data_dir, depth_path_file), 'r') as f:
        idx2depth_path = {
            '_'.join(depth_path.split('/')[1:-2]): join(data_dir, depth_path.replace('\n', ''))
            for depth_path in f.readlines()
        }

    rgb_path_file = 'train_rgb.txt' if train_flag else 'test_rgb.txt'
    with open(join(data_dir, rgb_path_file), 'r') as f:
        idx2rgb_path = {
            '_'.join(rgb_path.split('/')[1:-2]): join(data_dir, rgb_path.replace('\n', ''))
            for rgb_path in f.readlines()
        }

    gt_path_file = 'train_label.txt' if train_flag else 'test_label.txt'
    with open(join(data_dir, gt_path_file), 'r') as f:
        idx2gt_path = {
            '_'.join(gt_path.split('/')[1:-2]): join(data_dir, gt_path.replace('\n', ''))
            for gt_path in f.readlines()
        }
    return idx2depth_path, idx2rgb_path, idx2gt_path


class SunRGBDDataset(BaseSAMDataset):
    semantic_classes = [
        'void',
        'wall',
        'floor',
        'cabinet',
        'bed',
        'chair',
        'sofa',
        'table',
        'door',
        'window',
        'bookshelf',
        'picture',
        'counter',
        'blinds',
        'desk',
        'shelves',
        'curtain',
        'dresser',
        'pillow',
        'mirror',
        'floor mat',
        'clothes',
        'ceiling',
        'books',
        'fridge',
        'tv',
        'paper',
        'towel',
        'shower curtain',
        'box',
        'whiteboard',
        'person',
        'night stand',
        'toilet',
        'sink',
        'lamp',
        'bathtub',
        'bag'
    ]

    def __init__(
            self,
            is_train: bool,
            data_dir: str = f"{DATA_ROOT}/sunrgbd",
            image_type: str = 'rgb_images',
            transforms: Union[Dict, Compose, None] = None,
            **prompt_args
    ):
        if image_type not in ['rgb_images', 'depth_images', 'depth_rgb_images']:
            raise ValueError("Invalid image_type! Must be one of 'rgb_images', 'depth_images', 'depth_rgb_images'.")
        self.image_type = image_type

        if is_train:
            image_path_file, depth_path_file, gt_path_file = 'train_rgb.txt', 'train_depth.txt', 'train_label.txt'
        else:
            image_path_file, depth_path_file, gt_path_file = 'test_rgb.txt', 'test_depth.txt', 'test_label.txt'

        data_dict = {}
        with open(join(data_dir, image_path_file), 'r') as f:
            for image_path in f.readlines():
                image_path = image_path.replace('\n', '')
                index_name = '_'.join(image_path.split('/')[1:-2])
                data_dict[index_name] = dict(image_path=join(data_dir, image_path))
        with open(join(data_dir, depth_path_file), 'r') as f:
            for depth_path in f.readlines():
                depth_path = depth_path.replace('\n', '')
                index_name = '_'.join(depth_path.split('/')[1:-2])
                data_dict[index_name].update(depth_path=join(data_dir, depth_path))
        with open(join(data_dir, gt_path_file), 'r') as f:
            for gt_path in f.readlines():
                gt_path = gt_path.replace('\n', '')
                index_name = '_'.join(gt_path.split('/')[1:-2])
                data_dict[index_name].update(gt_mask_path=join(data_dir, gt_path))

        super(SunRGBDDataset, self).__init__(
            data_dict=data_dict, is_train=is_train, label_threshold=None, **prompt_args
        )

        # register the aligned transforms for RGB-Depth-MASK triples
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
        rgb_image = ret_dict['images'].permute(1, 2, 0).numpy()
        gt_mask = ret_dict['gt_masks'].numpy()
        depth_image = read_depth_from_disk(
            depth_path=self.data_dict[ret_dict['index_name']]["depth_path"], scale_factor=1000
        )

        # aligned transforms for the triple data
        if self.transforms is not None:
            transformed = self.transforms(image=rgb_image, depth=depth_image, mask=gt_mask)
            rgb_image, depth_image, gt_mask = transformed["image"], transformed['depth'], transformed['mask']

        if self.image_type == 'rgb_images':
            image_shape = (rgb_image.shape[0], rgb_image.shape[1])
        elif self.image_type in ['depth_images', 'depth_rgb_images']:
            image_shape = (depth_image.shape[0], depth_image.shape[1])
        else:
            raise RuntimeError(f"Meet wrong image type during training: {self.image_type}!")
        point_coords, box_coords, noisy_object_masks, object_masks, object_classes = (
            self.get_prompts_from_gt_masks(gt_mask=gt_mask, image_shape=image_shape)
        )

        # register transformed triple data (or the original ones if no transforms are specified)
        depth_rgb_image = nonrgb_to_rgb(depth_image)
        ret_dict.update(
            rgb_images=torch.from_numpy(rgb_image).to(torch.float32).permute(2, 0, 1),
            depth_images=torch.from_numpy(depth_image).to(torch.float32).permute(2, 0, 1),
            depth_rgb_images=torch.from_numpy(depth_rgb_image).to(torch.float32).permute(2, 0, 1),
            gt_masks=torch.from_numpy(gt_mask),
            point_coords=point_coords,
            box_coords=box_coords,
            noisy_object_masks=noisy_object_masks,
            object_masks=object_masks,
            object_classes=object_classes
        )
        ret_dict['images'] = ret_dict[self.image_type]
        return ret_dict


@TRANSFER_DATASETS.register("sunrgbd")
class SunRGBDTransfer(SunRGBDDataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="depth_images", **init_kwargs)


@FUSION_DATASETS.register("sunrgbd")
class SunRGBDFusion(SunRGBDDataset):
    def __init__(self, **init_kwargs):
        super().__init__(image_type="rgb_images", **init_kwargs)
