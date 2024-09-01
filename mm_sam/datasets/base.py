import random
from typing import Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mm_sam.datasets.gen_prompts import generate_prompts_from_semantic_mask
from utilbox.data_load.loader_utils import batch_list_to_dict
from utilbox.data_load.read_utils import read_image_as_rgb_from_disk, read_greyscale_mask_from_disk


class BaseSAMDataset(Dataset):
    # shared default arguments of transforms for all children classes
    transforms_default_args = {
        'VerticalFlip': {
            'p': 0.5
        },
        'HorizontalFlip': {
            'p': 0.5
        },
        'RandomRotate90': {
            'p': 0.5
        },
        'Transpose': {
            'p': 0.5
        },
        'RandomBrightness': {
            'limit': 0.1,
            'p': 1.0
        },
        'RandomContrast': {
            'limit': 0.1,
            'p': 1.0
        },
        'RandomCrop': {
            'scale': [0.1, 1.0],
            'p': 1.0
        },
        'RandomResizedCrop': {
            'scale': [0.1, 1.0],
            'p': 1.0
        }
    }

    semantic_classes: Optional[List] = None

    def __init__(
            self,
            data_dict: Dict,
            is_train: bool,
            used_prompts: Union[str, List[str], None] = None,
            label_threshold: Optional[int] = 128,
            **prompt_kwargs
    ):
        self.data_dict = data_dict
        self.index_name_list = list(self.data_dict.keys())

        self.is_train = is_train
        self.label_threshold = label_threshold
        self.prompt_kwargs = prompt_kwargs

        # default to only use the bounding boxes as the prompts
        if used_prompts is None:
            used_prompts = ['box']
        if isinstance(used_prompts, str):
            used_prompts = [used_prompts]
        # only `box` is used for prompts during testing
        if not self.is_train:
            used_prompts = ['box']
        for prompt in used_prompts:
            assert prompt in ['point', 'box', 'mask'], (
                f"{prompt} is not a valid prompt! Must be one of 'point', 'box', and 'mask'."
            )
        self.used_prompts = used_prompts

    def __len__(self):
        return len(self.index_name_list)

    def __getitem__(self, index):
        index_name = self.index_name_list[index]
        index_dict = self.data_dict[index_name]
        assert "image_path" in index_dict and "gt_mask_path" in index_dict, (
            "The data dict of a single instance should contain the key `image_path` for the RGB image file and "
            "`gt_mask_path` for the GT mask file."
        )
        image = self.get_image_by_path(index_dict['image_path'])
        gt_mask = self.get_gt_by_path(index_dict['gt_mask_path'])

        # (H, W, C) -> (C, H, W), datatype must be float32 for image data (cannot be uint or int)
        image = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1)
        # gt_mask keeps the original resolution since loss is not calculated on the 1024x1024 scale
        gt_mask = torch.from_numpy(gt_mask)
        return dict(
            images=image, gt_masks=gt_mask, index_name=index_name
        )

    def get_image_by_path(self, image_path: str):
        return read_image_as_rgb_from_disk(image_path)

    def get_gt_by_path(self, gt_path: str):
        return read_greyscale_mask_from_disk(gt_path, self.label_threshold)

    def get_prompts_from_gt_masks(self, gt_mask: np.ndarray, image_shape: Optional[Tuple[int, int]] = None):
        tgt_prompts = [random.choice(self.used_prompts)]
        point_coords, box_coords, noisy_object_masks, object_masks, object_classes = (
            generate_prompts_from_semantic_mask(
                image_shape=image_shape, gt_mask=gt_mask, tgt_prompts=tgt_prompts, **self.prompt_kwargs
            )
        )
        return point_coords, box_coords, noisy_object_masks, object_masks, object_classes

    @classmethod
    def collate_fn(cls, batch: Sequence[Dict]) -> Dict:
        batch_dict = batch_list_to_dict(batch)
        batch_dict['object_masks'] = [
            torch.from_numpy(item) if item is not None else None for item in batch_dict['object_masks']
        ]

        # pad the prompt points with placeholders (-1) to make sure that each prompt has the same number of points
        point_coords, point_labels = [], []
        for item in batch_dict['point_coords']:
            # give a None value to the images without any prompt points
            if item is None:
                point_coords.append(None)
                point_labels.append(None)
            # all the labels of prompt points are either foreground points (label=1) or placeholder (label=-1)
            else:
                _point_coords, _point_labels = item, []
                max_num_coords = max(len(_p_c) for _p_c in _point_coords)
                for _p_c in _point_coords:
                    _point_labels.append([1 for _ in _p_c])

                    curr_num_coords = len(_p_c)
                    if curr_num_coords < max_num_coords:
                        _p_c.extend([[0, 0] for _ in range(max_num_coords - curr_num_coords)])
                        _point_labels[-1].extend([-1 for _ in range(max_num_coords - curr_num_coords)])

                point_coords.append(torch.FloatTensor(_point_coords))
                point_labels.append(torch.LongTensor(_point_labels))
        batch_dict['point_coords'] = point_coords
        batch_dict['point_labels'] = point_labels

        # give a None value to the images without any prompt boxes
        batch_dict['box_coords'] = [
            torch.FloatTensor(item) if item is not None else None for item in batch_dict['box_coords']
        ]

        # we don't stack all the image and ground-truth tensors together to deal with different image sizes
        return batch_dict
