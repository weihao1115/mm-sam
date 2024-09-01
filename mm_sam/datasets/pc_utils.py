from typing import Tuple, List

import numpy as np


def proj_3d_scan_to_2d_img(
        proj_yx: np.ndarray, scan_data: np.ndarray, img_size: Tuple,
        sem_labels: np.ndarray = None, inst_labels: np.ndarray = None,
        foreground_only: bool = True, ignore_idx: int = 255, background_idx: int = 0, placeholder_idx: int = 0
):

    # Vectorized version of the record_instance_id function
    def record_instance_id(sem_label_ids):
        # Check whether the input semantic labels are ignored or background
        not_ignored = sem_label_ids != ignore_idx
        if foreground_only:
            not_background = sem_label_ids != background_idx
        else:
            not_background = True
        return not_ignored & not_background

    # loop the corresponding 2d pixel of each 3d point
    proj_img = np.ones((img_size[0], img_size[1], scan_data.shape[-1]), dtype=np.float32) * placeholder_idx
    sem_mask = None
    if sem_labels is not None:
        sem_mask = np.ones((img_size[0], img_size[1]), dtype=np.uint8) * ignore_idx
    inst_mask = None
    if inst_labels is not None:
        inst_mask = np.ones((img_size[0], img_size[1]), dtype=np.uint8) * ignore_idx

    # Create boolean masks for valid row and column indices
    valid_rows = (proj_yx[:, 0] >= 0) & (proj_yx[:, 0] < img_size[0])
    valid_cols = (proj_yx[:, 1] >= 0) & (proj_yx[:, 1] < img_size[1])
    valid_indices = valid_rows & valid_cols

    # Filter proj_yx using the valid indices
    # Note: there are repeated coordinates in filtered_proj_yx
    filtered_proj_yx = proj_yx[valid_indices]

    # Assign values to proj_img and sem_mask using the valid indices
    proj_img[filtered_proj_yx[:, 0], filtered_proj_yx[:, 1]] = scan_data[valid_indices]
    if sem_mask is not None:
        sem_mask[filtered_proj_yx[:, 0], filtered_proj_yx[:, 1]] = sem_labels[valid_indices]

    # For inst_mask, we need to check whether to record instance IDs
    if inst_mask is not None:
        # Use the vectorized function to create a boolean mask
        record_mask = record_instance_id(sem_labels[valid_indices])

        # Apply this mask to double filter proj_yx
        filtered_proj_yx_inst = filtered_proj_yx[record_mask]

        # Assign instance labels to inst_mask
        inst_mask[filtered_proj_yx_inst[:, 0], filtered_proj_yx_inst[:, 1]] = inst_labels[valid_indices][record_mask]

    return proj_img, sem_mask, inst_mask


def get_prompts_by_sem_inst_masks(
    sem_mask: np.ndarray, inst_mask: np.ndarray, tgt_prompts: List[str] = None,
    area_threshold: int = 4, prompt_point_num: int = 1, foreground_only: bool = True,
    ignore_idx: int = 255, background_idx: int = 0
):
    if tgt_prompts is None: tgt_prompts = []
    for prompt in tgt_prompts: assert prompt in ['point', 'box'], f'Invalid prompt: {prompt}!'

    # process the stacks (extract box coords from instance labels)
    point_coords, box_coords, object_masks, object_classes = [], [], [], []
    ignore_flags = sem_mask == ignore_idx
    mask_height, mask_width = sem_mask.shape[0], sem_mask.shape[1]

    inst_id_list = np.unique(inst_mask)
    for inst_id in inst_id_list:
        # Note: for instance id, 0 does not denote the background **BUT** the first object of a given semantic class!
        # skip the placeholder (ignore_idx)
        if inst_id == ignore_idx:
            continue

        x_axis, y_axis = np.meshgrid(np.arange(mask_width), np.arange(mask_height))
        inst_flags = inst_mask == inst_id
        inst_sem_labels = sem_mask[inst_flags]
        inst_sem_label_list = np.unique(inst_sem_labels)
        # in case that multiple objects of different semantic classes share the same instance id
        for sem_label_idx in inst_sem_label_list:
            if sem_label_idx == ignore_idx:
                continue
            if foreground_only and sem_label_idx == background_idx:
                continue

            sem_label_idx = sem_label_idx.item()
            inst_sem_flags = inst_flags & (sem_mask == sem_label_idx)
            inst_x_coords, inst_y_coords = x_axis[inst_sem_flags], y_axis[inst_sem_flags]

            # filter out the instance whose area is lower than the threshold
            inst_area = len(inst_x_coords)
            if inst_area < area_threshold:
                continue

            # make object_mask from inst_sem_flags and sem_label_idx
            _object_mask = np.ones_like(sem_mask, dtype=np.uint8) * background_idx
            _object_mask[ignore_flags] = ignore_idx
            _object_mask[inst_sem_flags] = 1
            object_masks.append(_object_mask)
            object_classes.append(sem_label_idx)

            # convert np.int64 to normal int for json serialization
            if 'point' in tgt_prompts:
                random_idxs = np.random.permutation(inst_x_coords.shape[0])[:prompt_point_num]
                point_coords.append(
                    [[int(inst_x_coords[idx]), int(inst_y_coords[idx])] for idx in random_idxs]
                )
            if 'box' in tgt_prompts:
                min_x = max(inst_x_coords.min().item() - 1, 0)
                min_y = max(inst_y_coords.min().item() - 1, 0)
                max_x = min(inst_x_coords.max().item() + 1, mask_width - 1)
                max_y = min(inst_y_coords.max().item() + 1, mask_height - 1)
                box_coords.append([min_x, min_y, max_x, max_y])

    # containing only 0 means no foreground object in the image
    if len(point_coords) == 0: point_coords = None
    if len(box_coords) == 0: box_coords = None
    if len(object_masks) == 0:
        object_masks.append(np.zeros(shape=(mask_height, mask_width), dtype=np.float32))
        object_classes.append(0)
    object_masks = np.stack(object_masks, axis=0)
    return point_coords, box_coords, object_masks, object_classes
