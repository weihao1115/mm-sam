from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from utilbox.demo_vis.vis_utils import nonrgb_to_rgb


def read_image_as_rgb_from_disk(image_path: str, return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    if image_path.endswith('npy'):
        image = np.load(image_path)
    else:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32)

    # make sure that img ranges from 0. to 255.
    if image.max() <= 1.0:
        image = np.round(image * 255.)

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, None], repeats=3, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)
    elif len(image.shape) != 3 and image.shape[-1] != 3:
        raise RuntimeError(f'Wrong image shape: {image.shape}. It should be either [H, W] or [H, W, 1] or [H, W, 3]!')

    if return_tensor:
        image = torch.from_numpy(image)
    return image


def read_greyscale_mask_from_disk(
        mask_path: str, label_threshold: Optional[int]= None, return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    if mask_path.endswith('npy'):
        gt_mask = np.load(mask_path)
    else:
        gt_mask = Image.open(mask_path).convert('L')
        gt_mask = np.array(gt_mask, dtype=np.float32)

    if len(gt_mask.shape) == 3:
        assert gt_mask.shape[-1] == 1, f"Wrong mask shape: {gt_mask.shape}. Should be (H, W) or (H, W, 1)!"
        gt_mask = gt_mask[:, :, 0]
    elif len(gt_mask.shape) != 2:
        raise RuntimeError(f"Wrong mask shape: {gt_mask.shape}. Should be (H, W) or (H, W, 1)!")

    # For the 0-255 mask, discretize its values into 0.0 or 1.0
    if label_threshold is not None:
        # make sure that gt_mask ranges from 0. to 255.
        if gt_mask.max() <= 1.0:
            gt_mask = gt_mask * 255.
        gt_mask = np.where(gt_mask > label_threshold, 1.0, 0.0)

    if return_tensor:
        gt_mask = torch.from_numpy(gt_mask)
    return gt_mask


def read_depth_from_disk(
        depth_path: str, scale_factor: Optional[int] = 1000, use_disparity: bool = True,
        to_rgb: bool = False, return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    depth = Image.open(depth_path)
    depth = np.array(depth, dtype=np.float32)

    if scale_factor is not None:
        depth /= scale_factor
    # convert depth from the original scale to the disparity scale
    if use_disparity:
        depth = 1 / depth
    if len(depth.shape) == 2:
        depth = depth[:, :, None]

    # return the false-color RGB depth images (mainly for SAM zero-shot)
    if to_rgb:
        depth = nonrgb_to_rgb(depth)
    if return_tensor:
        depth = torch.from_numpy(depth)
    return depth
