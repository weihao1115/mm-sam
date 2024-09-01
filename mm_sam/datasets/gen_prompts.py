import random
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def find_objects_from_mask(
        mask: np.ndarray,
        relative_threshold: bool = True,
        area_threshold: int = 20,
        relative_threshold_ratio: float = 0.001,
        connectivity: int = 8
):
    # from https://github.com/KyanChen/RSPrompter/blob/cky/tools/ins_seg/dataset_converters/whu_building_convert.py
    # Here, we only consider the mask values 1.0 as positive class, i.e., 255 pixel values
    object_num, objects_im, stats, centroids = cv2.connectedComponentsWithStats(
        image=mask.astype(np.uint8), connectivity=connectivity)

    # if no foreground object is found, a tuple of None is returned
    if object_num < 2:
        return None, None

    object_areas, object_indices_all, object_masks = [], [], []
    for i in range(1, object_num):
        object_mask = (objects_im == i).astype(np.float32)
        object_masks.append(object_mask)

        object_indices = np.argwhere(object_mask)
        object_areas.append(len(object_indices))
        object_indices_all.append(object_indices)

    # update area_threshold if relative_threshold is set
    if relative_threshold and len(object_indices_all) > 0:
        max_area = max(object_areas)
        area_threshold = max(max_area * relative_threshold_ratio, area_threshold)

    valid_objects = [i for i, o_a in enumerate(object_areas) if o_a > area_threshold]
    # if no foreground object is valid (area larger than the threshold), a tuple of None is returned
    if len(valid_objects) == 0:
        return None, None

    object_indices_all = [object_indices_all[i] for i in valid_objects]
    object_masks = [object_masks[i] for i in valid_objects]
    return object_indices_all, np.stack(object_masks, axis=0)


def find_random_points_in_objects(
        object_regions: List[np.ndarray],
        prompt_point_num: int = 1,
        random_num_prompt: bool = True
):
    # only select prompt points for the regions whose areas are larger than the threshold
    points = []
    for object_indices in object_regions:
        if random_num_prompt:
            # we randomly select a random number for each object during training
            _prompt_point_num = random.randint(1, prompt_point_num)
        else:
            _prompt_point_num = prompt_point_num

        # randomly select the given number of prompt points from each region
        random_idxs = np.random.permutation(object_indices.shape[0])[:_prompt_point_num]
        object_points = [[int(object_indices[idx][1]), int(object_indices[idx][0])] for idx in random_idxs]
        points.append(object_points)
    return points


def find_bound_box_on_objects(object_regions: List[np.ndarray]):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if len(object_regions) == 0:
        return None

    boxes = []
    for object_indices in object_regions:
        y_max, x_max = object_indices[:, 0].max(), object_indices[:, 1].max()
        y_min, x_min = object_indices[:, 0].min(), object_indices[:, 1].min()
        # for each object, we append a 2-dim List for representing its bound box
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def make_noisy_mask_on_objects(object_masks, scale_factor: int = 8, noisy_mask_threshold: float = 0.5):
    """
        Add noise to mask input
        From Mask Transfiner https://github.com/SysCV/transfiner
    """

    def get_incoherent_mask(input_masks):
        mask = input_masks.float()
        h, w = input_masks.shape[-2:]

        mask_small = F.interpolate(mask, (h // scale_factor, w // scale_factor), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (256, 256), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue

    noisy_object_masks = []
    for i, o_m in enumerate(object_masks):
        o_m_256 = F.interpolate(torch.from_numpy(o_m[None, None, :]), (256, 256), mode='bilinear')

        mask_noise = torch.randn(o_m_256.shape) * 1.0
        inc_masks = get_incoherent_mask(o_m_256)
        o_m_256 = ((o_m_256 + mask_noise * inc_masks) > noisy_mask_threshold).float()

        if len(o_m_256.shape) == 4 and o_m_256.size(0) == 1:
            o_m_256 = o_m_256.squeeze(0)
        noisy_object_masks.append(o_m_256)

    noisy_object_masks = torch.stack(noisy_object_masks, dim=0)
    return noisy_object_masks


def generate_prompts_from_mask(
        gt_mask: np.ndarray,
        tgt_prompts: List[str],
        image_shape: Optional[Tuple[int, int]] = None,
        # prompt_kwargs
        object_connectivity: int = 8,
        area_threshold: int = 20,
        relative_threshold: bool = True,
        relative_threshold_ratio: float = 0.001,
        max_object_num: int = None,
        prompt_point_num: int = 1,
        ann_scale_factor: int = 8,
        noisy_mask_threshold: float = 0.5,
        max_side_ratio: float = None
):
    if image_shape is None:
        image_shape = (gt_mask.shape[-2], gt_mask.shape[-1])

    point_coords, box_coords, noisy_object_masks = None, None, None
    object_regions, object_masks = find_objects_from_mask(
        gt_mask, connectivity=object_connectivity,
        area_threshold=area_threshold, relative_threshold=relative_threshold,
        relative_threshold_ratio=relative_threshold_ratio
    )

    # for the case that image and gt_mask are in different shapes
    image2mask_h_ratio = image_shape[0] / gt_mask.shape[0]
    image2mask_w_ratio = image_shape[1] / gt_mask.shape[1]

    # skip prompt generation if no object is found in the gt_mask
    if object_regions is None:
        # since object_masks act as the label for training, we give one zero mask when there is no object
        object_masks = np.zeros(shape=(1, gt_mask.shape[0], gt_mask.shape[1]), dtype=np.float32)
    else:
        if max_object_num is not None and len(object_regions) > max_object_num:
            random_object_idxs = np.random.permutation(len(object_regions))[:max_object_num]
            object_regions = [object_regions[idx] for idx in random_object_idxs]
            object_masks = np.stack([object_masks[idx] for idx in random_object_idxs], axis=0)

        # filter the prompts whose box has side ratio larger than max_side_ratio
        _box_coords = find_bound_box_on_objects(object_regions)
        if max_side_ratio is not None:
            prompt_keep_flags = []
            for box in _box_coords:
                width, height = box[2] - box[0], box[3] - box[1]
                side_ratio = max(width, height) / min(width, height)
                prompt_keep_flags.append(side_ratio < max_side_ratio)
        else:
            prompt_keep_flags = [True] * len(_box_coords)

        # skip the current prompt generation if no box has side ratio lower than the threshold
        if True not in prompt_keep_flags:
            object_masks = np.zeros(shape=(1, gt_mask.shape[0], gt_mask.shape[1]), dtype=np.float32)
        else:
            object_masks = object_masks[prompt_keep_flags]

            if 'point' in tgt_prompts:
                point_coords = find_random_points_in_objects(
                    object_regions, prompt_point_num=prompt_point_num
                )
                point_coords = [p_coords for p_idx, p_coords in enumerate(point_coords) if prompt_keep_flags[p_idx]]
                for points in point_coords:
                    for point in points:
                        point[0] = int(point[0] * image2mask_w_ratio)
                        point[1] = int(point[1] * image2mask_h_ratio)

            if 'box' in tgt_prompts:
                box_coords = _box_coords
                box_coords = [b_coords for b_idx, b_coords in enumerate(box_coords) if prompt_keep_flags[b_idx]]
                for box in box_coords:
                    box[0], box[2] = int(box[0] * image2mask_w_ratio), int(box[2] * image2mask_w_ratio)
                    box[1], box[3] = int(box[1] * image2mask_h_ratio), int(box[3] * image2mask_h_ratio)

            if 'mask' in tgt_prompts:
                noisy_object_masks = make_noisy_mask_on_objects(
                    object_masks=object_masks, scale_factor=ann_scale_factor,
                    noisy_mask_threshold=noisy_mask_threshold
                )
                noisy_object_masks = noisy_object_masks[prompt_keep_flags]
                # don't need to deal with the case that image and gt_mask have different shapes for noisy masks
                # since the output noisy_object_masks here is always 256x256 for SAM compatibility

    return point_coords, box_coords, noisy_object_masks, object_masks


def generate_prompts_from_semantic_mask(
        gt_mask: np.ndarray, tgt_prompts: List[str], image_shape: Optional[Tuple[int, int]] = None, **prompt_kwargs
):
    if image_shape is None:
        image_shape = (gt_mask.shape[-2], gt_mask.shape[-1])

    point_coords, box_coords, noisy_object_masks, object_masks, object_classes = [], [], [], [], []
    class_list = np.unique(gt_mask).astype(np.int64).tolist()
    for class_idx in class_list:
        if class_idx == 0:
            continue
        point_coords_class, box_coords_class, noisy_object_masks_class, object_masks_class = \
            generate_prompts_from_mask(
                image_shape=image_shape, gt_mask=np.where(gt_mask == class_idx, 1., 0.),
                tgt_prompts=tgt_prompts, **prompt_kwargs
            )

        if point_coords_class is not None or box_coords_class is not None or noisy_object_masks_class is not None:
            object_classes.extend([class_idx for _ in object_masks_class])
            object_masks.extend([obj_mask[None, :] for obj_mask in object_masks_class])
            if point_coords_class is not None:
                point_coords.extend(point_coords_class)
            if box_coords_class is not None:
                box_coords.extend(box_coords_class)
            if noisy_object_masks_class is not None:
                noisy_object_masks.append(noisy_object_masks_class)

    if len(object_masks) == 0:
        object_masks.append(np.zeros(shape=(1, gt_mask.shape[-2], gt_mask.shape[-1]), dtype=np.float32))
        object_classes.append(0)
    object_masks = np.concatenate(object_masks, axis=0)

    if len(point_coords) == 0:
        point_coords = None
    if len(box_coords) == 0:
        box_coords = None
    noisy_object_masks = torch.cat(noisy_object_masks, dim=0) if len(noisy_object_masks) > 0 else None
    return point_coords, box_coords, noisy_object_masks, object_masks, object_classes


def generate_prompts_from_object_masks(
        object_masks: np.ndarray,
        tgt_prompts: List[str],
        image_shape: Optional[Tuple[int, int]] = None,
        # prompt_kwargs
        area_threshold: int = 0,
        max_object_num: int = None,
        prompt_point_num: int = 1,
        ann_scale_factor: int = 8,
        noisy_mask_threshold: float = 0.5
):
    if image_shape is None:
        image_shape = (object_masks.shape[-2], object_masks.shape[-1])

    object_regions = [np.argwhere(o_mask) for o_mask in object_masks]
    object_keep_flags = [len(obj_r) > area_threshold for obj_r in object_regions]
    object_regions = [obj_r for i, obj_r in enumerate(object_regions) if object_keep_flags[i]]
    object_masks = object_masks[object_keep_flags]

    if len(object_regions) == 0:
        return None, None, None, np.zeros(shape=(1, image_shape[0], image_shape[1]), dtype=object_masks.dtype)

    if max_object_num is not None and len(object_regions) > max_object_num:
        random_object_idxs = np.random.permutation(len(object_regions))[:max_object_num]
        object_regions = [object_regions[idx] for idx in random_object_idxs]
        object_masks = object_masks[random_object_idxs]

    point_coords, box_coords, noisy_object_masks = None, None, None
    # for the case that image and gt_mask are in different shapes
    image2mask_h_ratio = image_shape[0] / object_masks.shape[-2]
    image2mask_w_ratio = image_shape[1] / object_masks.shape[-1]

    if 'point' in tgt_prompts:
        point_coords = find_random_points_in_objects(
            object_regions, prompt_point_num=prompt_point_num
        )
        # some logics here to deal with the case that image2mask_h_ratio or image2mask_w_ratio is not one
        assert image2mask_h_ratio == 1 and image2mask_w_ratio == 1

    if 'box' in tgt_prompts:
        box_coords = find_bound_box_on_objects(object_regions)
        for box in box_coords:
            box[0], box[2] = int(box[0] * image2mask_w_ratio), int(box[2] * image2mask_w_ratio)
            box[1], box[3] = int(box[1] * image2mask_h_ratio), int(box[3] * image2mask_h_ratio)

    if 'mask' in tgt_prompts:
        noisy_object_masks = make_noisy_mask_on_objects(
            object_masks=object_masks, scale_factor=ann_scale_factor,
            noisy_mask_threshold=noisy_mask_threshold
        )
        # don't need to deal with the case that image and gt_mask have different shapes for noisy masks
        # since the output noisy_object_masks here is always 256x256 for SAM compatibility

    return point_coords, box_coords, noisy_object_masks, object_masks
