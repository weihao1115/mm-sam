import argparse
import json
import math
import os
from functools import partial
from os.path import join
from typing import List

import laspy
from tqdm import tqdm

from mm_sam.datasets.pc_utils import proj_3d_scan_to_2d_img
from utilbox.global_config import DATA_ROOT
from utilbox.demo_vis.vis_utils import visualize_nonrgb_image

import spectral.io.envi as envi
from osgeo import gdal
import cv2
import numpy as np
from PIL import Image

from utilbox.demo_vis.color_lib import SEMANTIC_KITTI_RGB_COLOR_ARRAY


def get_color_mapping_for_pc_mask():
    # vis all classes
    label_mapping_all = np.zeros(256, dtype=np.int64)
    label_mapping_all[1] = 1
    label_mapping_all[2] = 2
    label_mapping_all[6] = 3
    label_mapping_all[7] = 4
    label_mapping_all[17] = 5
    color_mapping_all = SEMANTIC_KITTI_RGB_COLOR_ARRAY[label_mapping_all]

    # only vis building-related classes (6 & 17)
    label_mapping_building = np.zeros(256, dtype=np.int64)
    label_mapping_building[6] = 1
    label_mapping_building[17] = 1
    color_mapping_building = SEMANTIC_KITTI_RGB_COLOR_ARRAY[label_mapping_building]

    return color_mapping_all, color_mapping_building


def get_color_mapping_for_rgb_mask():
    # vis all classes
    label_mapping_all = np.zeros(256, dtype=np.int64)
    for i in range(21):
        label_mapping_all[i] = i
    color_mapping_all = SEMANTIC_KITTI_RGB_COLOR_ARRAY[label_mapping_all]

    # only vis building-related classes (8 & 9)
    label_mapping_building = np.zeros(256, dtype=np.int64)
    label_mapping_building[8] = 1
    label_mapping_building[9] = 1
    color_mapping_building = SEMANTIC_KITTI_RGB_COLOR_ARRAY[label_mapping_building]

    return color_mapping_all, color_mapping_building


def prepare_test_data(rgb_dir, gt_dir, test_vis_save_dir):
    # -------------------- RGB Preparation Start -------------------- #
    test_min_y, test_rgb_image_stack = 3289689, []
    for test_min_x in [272652, 273248]:
        rgb_dataset = gdal.Open(join(rgb_dir, f'UH_NAD83_{test_min_x}_{test_min_y}.tif'))
        # (12020, 11920, 3)
        test_rgb_image_stack.append(np.transpose(rgb_dataset.ReadAsArray(), (1, 2, 0)))

    # shape (12020, 23840, 3)
    test_rgb_image = np.concatenate(test_rgb_image_stack, axis=1)
    rgb_tile_height, rgb_tile_width = test_rgb_image.shape[0], int(test_rgb_image.shape[1] / 2)
    # -------------------- RGB Preparation End -------------------- #

    # -------------------- GT Mask Preparation Start -------------------- #
    # manually-made pixel-wise GT mask
    gt_mask_dataset = gdal.Open(join(gt_dir, 'test_gt_mask_ori_scale.png'))
    # shape (12020, 23840)
    gt_mask = gt_mask_dataset.ReadAsArray()
    gt_tile_height, gt_tile_width = gt_mask.shape[0], int(gt_mask.shape[1] / 2)
    # -------------------- GT Mask Preparation End -------------------- #

    # -------------------- Test Data Visualization Start -------------------- #
    # visualize test data for checking
    test_rgb_image_vis = cv2.resize(
        test_rgb_image, dsize=(int(rgb_tile_width * 2 / 10), int(rgb_tile_height / 10)),
        interpolation=cv2.INTER_LINEAR
    )
    Image.fromarray(
        test_rgb_image_vis.astype(np.uint8)
    ).save(join(test_vis_save_dir, 'test_rgb_image.png'))

    color_mapping_all, color_mapping_building = get_color_mapping_for_rgb_mask()
    gt_mask_rgb_all = color_mapping_all[gt_mask]
    Image.fromarray(
        gt_mask_rgb_all.astype(np.uint8)
    ).save(join(test_vis_save_dir, 'test_gt_mask_all.png'))
    gt_mask_rgb_building = color_mapping_building[gt_mask]
    Image.fromarray(
        gt_mask_rgb_building.astype(np.uint8)
    ).save(join(test_vis_save_dir, f'test_gt_mask_building.png'))
    # -------------------- Test Data Visualization End -------------------- #
    return test_rgb_image, (rgb_tile_height, rgb_tile_width), gt_mask, (gt_tile_height, gt_tile_width)


def get_rotate_kwargs(
        angle_list, scale_list, ori_height, ori_width, interp_flag=cv2.INTER_LINEAR, border_value: int = 0
):
    rotate_kwargs_dict = {}
    for angle in angle_list:
        angle_key = f'{angle:d}'
        if angle_key not in rotate_kwargs_dict.keys():
            rotate_kwargs_dict[angle_key] = {}

        for scale in scale_list:
            scale_key = f'{scale:.2f}'
            if angle == 0:
                rotate_kwargs_dict[angle_key][scale_key] = None
            else:
                scaled_height, scaled_width = int(ori_height * scale), int(ori_width * scale)
                center = (scaled_width // 2, scaled_height // 2)

                sin_angle = np.abs(np.sin(np.radians(angle)))
                cos_angle = np.abs(np.cos(np.radians(angle)))
                new_width = int((scaled_height * sin_angle) + (scaled_width * cos_angle))
                new_height = int((scaled_height * cos_angle) + (scaled_width * sin_angle))

                rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                rot_mat[0, 2] += (new_width / 2) - center[0]
                rot_mat[1, 2] += (new_height / 2) - center[1]

                rotate_kwargs_dict[angle_key][scale_key] = dict(
                    M=rot_mat, dsize=(new_width, new_height),
                    flags=interp_flag, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
                )
    return rotate_kwargs_dict


def prepare_raster_data(raster_dir):
    raster_stack = []
    for i_channel in [1, 2, 3]:
        raster_dataset = gdal.Open(
            join(raster_dir, f'Intensity_C{i_channel}', f'UH17_GI{i_channel}F051.tif')
        )

        # shape (2404, 8344), 601x596 tiles are arranged in two rows and 7 columns
        raster_array = raster_dataset.ReadAsArray()
        # the value larger than the maximum is the original placeholder for the empty pixels
        band_max = raster_dataset.GetRasterBand(1).GetStatistics(True, True)[1]
        invalid_flags = raster_array > band_max
        raster_array[invalid_flags] = 0
        raster_stack.append(raster_array)
    # shape (2404, 8344, 3)
    raster_image = np.stack(raster_stack, axis=-1)
    raster_tile_height, raster_tile_width = int(raster_image.shape[0] / 2), int(raster_image.shape[1] / 7)
    return raster_image, (raster_tile_height, raster_tile_width)


def prepare_pc_data(
        pc_dir, raster_dir, raster_image_height, raster_image_width
):
    # geo transforms parameters of the .tif raster tile
    # all three channels are the same, so the first one is used
    raster_dataset = gdal.Open(join(raster_dir, f'Intensity_C1', f'UH17_GI1F051.tif'))
    raster_geo_transform = raster_dataset.GetGeoTransform()
    raster_geo_y_start, raster_geo_y_stride = raster_geo_transform[3], raster_geo_transform[5]
    raster_geo_x_start, raster_geo_x_stride = raster_geo_transform[0], raster_geo_transform[1]

    # loop tiles in each row
    grid_yx_tile_stack, geo_xyz_tile_stack, sem_labels_tile_stack = [], [], []
    for min_y in [3290290, 3289689]:
        # loop tiles in each column in the given row
        for min_x in [271460, 272056, 272652, 273248, 273844, 274440, 275036]:
            with laspy.open(join(pc_dir, 'C123', f'{min_x}_{min_y}.las')) as fh:
                las = fh.read()
                # (N, 3) 3d geometric coords, (x, y, z)
                geo_xyz_tile = las.xyz
                # (N,) semantic labels in point cloud data (different from the one in TrainingGT)
                sem_labels_tile = las.classification.array

                # round xy coords to 0.5-basis range to align with the strides in raster_geo_transform
                round_xy_tile = np.round(geo_xyz_tile[:, :2] * 2) / 2
                grid_y_tile = (round_xy_tile[:, 1] - raster_geo_y_start) / raster_geo_y_stride
                grid_x_tile = (round_xy_tile[:, 0] - raster_geo_x_start) / raster_geo_x_stride
                grid_yx_tile = np.stack([grid_y_tile, grid_x_tile], axis=-1).astype(int)

                grid_yx_tile_stack.append(grid_yx_tile)
                geo_xyz_tile_stack.append(geo_xyz_tile)
                sem_labels_tile_stack.append(sem_labels_tile)

    # (2404, 8344, 3) & (2404, 8344)
    proj_xyz_image, sem_mask, _ = proj_3d_scan_to_2d_img(
        proj_yx=np.concatenate(grid_yx_tile_stack, axis=0),
        scan_data=np.concatenate(geo_xyz_tile_stack, axis=0),
        sem_labels=np.concatenate(sem_labels_tile_stack, axis=0),
        img_size=(raster_image_height, raster_image_width)
    )
    return proj_xyz_image, sem_mask


def test_split_dump(
        args, test_rgb_image, rgb_tile_shape, test_hsi_image, hsi_tile_shape,
        test_xyz3c_image, test_pc_sem_mask, raster_tile_shape, gt_mask, gt_tile_shape
):
    rgb_tile_height, rgb_tile_width = rgb_tile_shape
    hsi_tile_height, hsi_tile_width = hsi_tile_shape
    raster_tile_height, raster_tile_width = raster_tile_shape
    gt_tile_height, gt_tile_width = gt_tile_shape

    gt_unit_height, gt_unit_width = \
        int(gt_tile_height / args.unit_num_per_side), int(gt_tile_width / args.unit_num_per_side)
    hsi_unit_height, hsi_unit_width = \
        int(hsi_tile_height / args.unit_num_per_side), int(hsi_tile_width / args.unit_num_per_side)
    raster_unit_height, raster_unit_width = \
        int(raster_tile_height / args.unit_num_per_side), int(raster_tile_width / args.unit_num_per_side)
    rgb_unit_height, rgb_unit_width = \
        int(rgb_tile_height / args.unit_num_per_side), int(rgb_tile_width / args.unit_num_per_side)

    # loop each unit to create test data samples
    unit_row_num = int(gt_mask.shape[0] / gt_unit_height)
    unit_col_num = int(gt_mask.shape[1] / gt_unit_width)
    test_sample_dict = {}
    for row_idx in range(unit_row_num):
        rgb_unit_h_start, rgb_unit_h_end = row_idx * rgb_unit_height, (row_idx + 1) * rgb_unit_height
        hsi_unit_h_start, hsi_unit_h_end = row_idx * hsi_unit_height, (row_idx + 1) * hsi_unit_height
        raster_unit_h_start, raster_unit_h_end = row_idx * raster_unit_height, (row_idx + 1) * raster_unit_height
        gt_unit_h_start, gt_unit_h_end = row_idx * gt_unit_height, (row_idx + 1) * gt_unit_height

        for col_idx in range(unit_col_num):
            rgb_unit_w_start, rgb_unit_w_end = col_idx * rgb_unit_width, (col_idx + 1) * rgb_unit_width
            hsi_unit_w_start, hsi_unit_w_end = col_idx * hsi_unit_width, (col_idx + 1) * hsi_unit_width
            raster_unit_w_start, raster_unit_w_end = col_idx * raster_unit_width, (col_idx + 1) * raster_unit_width
            gt_unit_w_start, gt_unit_w_end = col_idx * gt_unit_width, (col_idx + 1) * gt_unit_width

            sample_name = f'row{row_idx}_col{col_idx}'
            test_sample_dict[sample_name] = dict(
                rgb_image=test_rgb_image[rgb_unit_h_start:rgb_unit_h_end, rgb_unit_w_start:rgb_unit_w_end],
                hsi_image=test_hsi_image[hsi_unit_h_start:hsi_unit_h_end, hsi_unit_w_start:hsi_unit_w_end],
                proj_xyz3c_image=test_xyz3c_image[
                    raster_unit_h_start:raster_unit_h_end, raster_unit_w_start:raster_unit_w_end
                ],
                pc_sem_mask=test_pc_sem_mask[
                    raster_unit_h_start:raster_unit_h_end, raster_unit_w_start:raster_unit_w_end
                ],
                gt_mask=gt_mask[gt_unit_h_start:gt_unit_h_end, gt_unit_w_start:gt_unit_w_end],
            )

    # save each test sample to a specific .npz file
    test_npz_save_dir = join(args.tgt_data_path, 'test')
    os.makedirs(test_npz_save_dir, exist_ok=True)
    test_metadata = {}
    for sample_name in test_sample_dict.keys():
        np.savez_compressed(
            join(test_npz_save_dir, f'{sample_name}.npz'), **test_sample_dict[sample_name]
        )
        # metadata records the relative path to tgt_data_path for portability
        test_metadata[sample_name] = dict(
            npz_path=join('test', f'{sample_name}.npz'),
            rgb_image_path=join('test', f'{sample_name}.npz::rgb_image'),
            hsi_image_path=join('test', f'{sample_name}.npz::hsi_image'),
            proj_xyz3c_image_path=join('test', f'{sample_name}.npz::proj_xyz3c_image'),
            pc_sem_mask_path=join('test', f'{sample_name}.npz::pc_sem_mask'),
            gt_mask_path=join('test', f'{sample_name}.npz::gt_mask')
        )
    return test_metadata


def scale_rotate_and_clip(
        tile_args_list: List, npz_save_dir, max_placeholder_ratio,
        rgb_unit_shape, hsi_unit_shape, raster_unit_shape
):
    rgb_unit_height, rgb_unit_width = rgb_unit_shape
    hsi_unit_height, hsi_unit_width = hsi_unit_shape
    raster_unit_height, raster_unit_width = raster_unit_shape

    metadata_dict = {}
    outter_pbar = tqdm(total=len(tile_args_list), desc='loop tiles', leave=False)
    rgb_unit_numel = None
    for tile_args in tile_args_list:
        tile_row_idx, tile_col_idx, angle, scale = tile_args[:4]
        train_rgb_image_tile, train_hsi_image_tile, train_xyz3c_image_tile, train_pc_sem_mask_tile = tile_args[4:-4]
        rgb_rotate_kwargs, hsi_rotate_kwargs, raster_rotate_kwargs, sem_mask_rotate_kwargs = tile_args[-4:]

        # scale the RGB, HSI, and PC images
        train_hsi_image_tile_scale = cv2.resize(
            train_hsi_image_tile,
            dsize=(int(train_hsi_image_tile.shape[1] * scale), int(train_hsi_image_tile.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR
        )
        train_xyz3c_image_tile_scale = cv2.resize(
            train_xyz3c_image_tile,
            dsize=(int(train_xyz3c_image_tile.shape[1] * scale), int(train_xyz3c_image_tile.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR
        )
        # nearest method should be used to interpolate the semantic mask
        train_pc_sem_mask_tile_scale = cv2.resize(
            train_pc_sem_mask_tile,
            dsize=(int(train_pc_sem_mask_tile.shape[1] * scale), int(train_pc_sem_mask_tile.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST_EXACT
        )
        train_rgb_image_tile_scale = cv2.resize(
            train_rgb_image_tile,
            dsize=(int(train_rgb_image_tile.shape[1] * scale), int(train_rgb_image_tile.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR
        )

        # rotate the scaled RGB and HSI images
        if angle == 0:
            train_hsi_image_tile_rotate = train_hsi_image_tile_scale
            train_xyz3c_image_tile_rotate = train_xyz3c_image_tile_scale
            train_pc_sem_mask_tile_rotate = train_pc_sem_mask_tile_scale
            train_rgb_image_tile_rotate = train_rgb_image_tile_scale
        else:
            train_hsi_image_tile_rotate = cv2.warpAffine(train_hsi_image_tile_scale, **hsi_rotate_kwargs)
            train_xyz3c_image_tile_rotate = cv2.warpAffine(train_xyz3c_image_tile_scale, **raster_rotate_kwargs)
            train_pc_sem_mask_tile_rotate = cv2.warpAffine(train_pc_sem_mask_tile_scale, **sem_mask_rotate_kwargs)
            train_rgb_image_tile_rotate = cv2.warpAffine(train_rgb_image_tile_scale, **rgb_rotate_kwargs)

        # ensure the unit numbers are valid
        unit_row_num = min(
            int(train_rgb_image_tile_rotate.shape[0] / rgb_unit_height),
            int(train_hsi_image_tile_rotate.shape[0] / hsi_unit_height),
            int(train_xyz3c_image_tile_rotate.shape[0] / raster_unit_height),
            int(train_pc_sem_mask_tile_rotate.shape[0] / raster_unit_height),
        )
        unit_col_num = min(
            int(train_rgb_image_tile_rotate.shape[1] / rgb_unit_width),
            int(train_hsi_image_tile_rotate.shape[1] / hsi_unit_width),
            int(train_xyz3c_image_tile_rotate.shape[1] / raster_unit_width),
            int(train_pc_sem_mask_tile_rotate.shape[1] / raster_unit_width),
        )
        inner_pbar = tqdm(total=unit_row_num * unit_col_num, desc='loop units', leave=False)
        for unit_row_idx in range(unit_row_num):
            rgb_unit_h_start, rgb_unit_h_end = \
                unit_row_idx * rgb_unit_height, (unit_row_idx + 1) * rgb_unit_height
            hsi_unit_h_start, hsi_unit_h_end = \
                unit_row_idx * hsi_unit_height, (unit_row_idx + 1) * hsi_unit_height
            raster_unit_h_start, raster_unit_h_end = \
                unit_row_idx * raster_unit_height, (unit_row_idx + 1) * raster_unit_height

            for unit_col_idx in range(unit_col_num):
                rgb_unit_w_start, rgb_unit_w_end = \
                    unit_col_idx * rgb_unit_width, (unit_col_idx + 1) * rgb_unit_width
                hsi_unit_w_start, hsi_unit_w_end = \
                    unit_col_idx * hsi_unit_width, (unit_col_idx + 1) * hsi_unit_width
                raster_unit_w_start, raster_unit_w_end = \
                    unit_col_idx * raster_unit_width, (unit_col_idx + 1) * raster_unit_width

                train_rgb_unit = train_rgb_image_tile_rotate[
                    rgb_unit_h_start:rgb_unit_h_end, rgb_unit_w_start:rgb_unit_w_end
                ]
                train_hsi_unit = train_hsi_image_tile_rotate[
                    hsi_unit_h_start:hsi_unit_h_end, hsi_unit_w_start:hsi_unit_w_end
                ]
                train_xyz3c_unit = train_xyz3c_image_tile_rotate[
                    raster_unit_h_start:raster_unit_h_end, raster_unit_w_start:raster_unit_w_end
                ]
                train_pc_sem_unit = train_pc_sem_mask_tile_rotate[
                    raster_unit_h_start:raster_unit_h_end, raster_unit_w_start:raster_unit_w_end
                ]
                # checkpoint the ratio of placeholders
                if rgb_unit_numel is None:
                    rgb_unit_numel = np.ones_like(train_rgb_unit).sum()
                placeholder_ratio = (train_rgb_unit == 0).sum() / rgb_unit_numel
                if placeholder_ratio < max_placeholder_ratio:
                    # save to .npz file
                    sample_name = f'trow{tile_row_idx}_tcol{tile_col_idx}_' \
                                  f'angle{angle}_scale{scale:.2f}_' \
                                  f'urow{unit_row_idx}_ucol{unit_col_idx}'
                    sample_npz_save_path = join(npz_save_dir, f'{sample_name}.npz')
                    np.savez_compressed(
                        sample_npz_save_path,
                        rgb_image=train_rgb_unit, hsi_image=train_hsi_unit,
                        proj_xyz3c_image=train_xyz3c_unit, proj_sem_mask=train_pc_sem_unit,
                    )
                    # record path to metadata
                    metadata_dict[sample_name] = dict(
                        npz_path=join('train', f'{sample_name}.npz'),
                        rgb_image_path=join('train', f'{sample_name}.npz::rgb_image'),
                        hsi_image_path=join('train', f'{sample_name}.npz::hsi_image'),
                        proj_xyz3c_image_path=join('train', f'{sample_name}.npz::proj_xyz3c_image'),
                        proj_sem_mask_path=join('train', f'{sample_name}.npz::proj_sem_mask'),
                    )

                inner_pbar.update(1)
                inner_step_info = \
                    f"urow: {unit_row_idx + 1}/{unit_row_num}. ucol: {unit_col_idx + 1}/{unit_col_num}."
                inner_pbar.set_postfix_str(inner_step_info)

        outter_pbar.update(1)
        outter_step_info = f"trow: {tile_row_idx + 1}/2. tcol: {tile_col_idx + 1}/7. angle: {angle}. scale: {scale}"
        outter_pbar.set_postfix_str(outter_step_info)
    return metadata_dict


def train_split_dump(
        xyz3c_image, pc_sem_mask, raster_tile_shape, hsi_image, hsi_tile_shape, rgb_dir, vis_save_dir, args
):
    train_hsi_image, train_xyz3c_image, train_pc_sem_mask = hsi_image.copy(), xyz3c_image.copy(), pc_sem_mask.copy()
    hsi_tile_height, hsi_tile_width = hsi_tile_shape
    raster_tile_height, raster_tile_width = raster_tile_shape
    # loop tiles in each row
    rgb_image_row_stack = []
    for row_idx, min_y in enumerate([3290290, 3289689]):
        hsi_row_start, hsi_row_end = row_idx * hsi_tile_height, (row_idx + 1) * hsi_tile_height
        raster_row_start, raster_row_end = row_idx * raster_tile_height, (row_idx + 1) * raster_tile_height

        # loop tiles in each column in the given row
        rgb_image_col_stack = []
        for col_idx, min_x in enumerate([271460, 272056, 272652, 273248, 273844, 274440, 275036]):
            hsi_col_start, hsi_col_end = col_idx * hsi_tile_width, (col_idx + 1) * hsi_tile_width
            raster_col_start, raster_col_end = col_idx * raster_tile_width, (col_idx + 1) * raster_tile_width

            # mask the test tiles by the placeholders (0, default padding value of cv2.warpAffine)
            if min_y == 3289689 and min_x in [272652, 273248]:
                train_hsi_image[hsi_row_start:hsi_row_end, hsi_col_start:hsi_col_end] = 0
                train_xyz3c_image[raster_row_start:raster_row_end, raster_col_start:raster_col_end] = 0
                train_pc_sem_mask[raster_row_start:raster_row_end, raster_col_start:raster_col_end] = 255
                # (12020, 11920, 3)
                rgb_image_col_stack.append(np.zeros_like(rgb_image_col_stack[-1]))
            # record the train tiles
            else:
                rgb_dataset = gdal.Open(join(rgb_dir, f'UH_NAD83_{min_x}_{min_y}.tif'))
                # (12020, 11920, 3)
                rgb_image_col_stack.append(np.transpose(rgb_dataset.ReadAsArray(), (1, 2, 0)))

        # (12020, 83440, 3)
        rgb_image_row_stack.append(np.concatenate(rgb_image_col_stack, axis=1))

    # (24040, 83440, 3)
    train_rgb_image = np.concatenate(rgb_image_row_stack, axis=0)
    rgb_tile_height, rgb_tile_width = int(train_rgb_image.shape[0] / 2), int(train_rgb_image.shape[1] / 7)
    # visualize the entire RGB image for checking
    train_rgb_image_vis = cv2.resize(
        train_rgb_image, dsize=(train_hsi_image.shape[1], train_hsi_image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    train_rgb_image_vis[train_rgb_image_vis < 0] = 0.0
    Image.fromarray(
        train_rgb_image_vis.astype(np.uint8)
    ).save(join(vis_save_dir, 'train_rgb_image.png'))

    # shape (1202, 4172, 3)
    train_hsi_image_vis = train_hsi_image[:, :, [15, 31, 47]]
    visualize_nonrgb_image(train_hsi_image_vis, vis_save_dir=vis_save_dir, vis_file_name='train_hsi_default_bands')
    # shape (1202, 4172, 3)
    train_xyz3c_image_vis = train_xyz3c_image[:, :, 3:]
    visualize_nonrgb_image(train_xyz3c_image_vis, vis_save_dir=vis_save_dir, vis_file_name='train_raster_image')

    # shape (1202, 4172)
    color_mapping_all, color_mapping_building = get_color_mapping_for_pc_mask()
    # vis all classes
    pc_sem_mask_rgb_all = color_mapping_all[train_pc_sem_mask]
    Image.fromarray(
        pc_sem_mask_rgb_all.astype(np.uint8)
    ).save(join(vis_save_dir, f'train_proj_sem_mask_all.png'))
    # only vis building-related classes (6 & 17)
    pc_sem_mask_rgb_building = color_mapping_building[train_pc_sem_mask]
    Image.fromarray(
        pc_sem_mask_rgb_building.astype(np.uint8)
    ).save(join(vis_save_dir, f'train_proj_sem_mask_building.png'))

    angle_list = [args.min_angle] if args.min_angle == args.max_angle else \
        np.arange(args.min_angle, args.max_angle + args.angle_stride, args.angle_stride)
    scale_list = [args.min_scale] if args.min_scale == args.max_scale else \
        np.arange(args.min_scale, args.max_scale + args.scale_stride, args.scale_stride)
    # the rotation is done tile-wise!
    rgb_rotate_kwargs_dict = get_rotate_kwargs(
        angle_list, scale_list, ori_height=rgb_tile_height, ori_width=rgb_tile_width
    )
    hsi_rotate_kwargs_dict = get_rotate_kwargs(
        angle_list, scale_list, ori_height=hsi_tile_height, ori_width=hsi_tile_width
    )
    raster_rotate_kwargs_dict = get_rotate_kwargs(
        angle_list, scale_list, ori_height=raster_tile_height, ori_width=raster_tile_width
    )
    sem_mask_rotate_kwargs_dict = get_rotate_kwargs(
        angle_list, scale_list, ori_height=raster_tile_height, ori_width=raster_tile_width,
        interp_flag=cv2.INTER_NEAREST,  # cv2.INTER_NEAREST_EXACT is not supported by cv2.warpAffine()
    )
    # the shape of sample units is decided by the original image shape (not consider scale)
    hsi_unit_height, hsi_unit_width = \
        int(hsi_tile_height / args.unit_num_per_side), int(hsi_tile_width / args.unit_num_per_side)
    raster_unit_height, raster_unit_width = \
        int(raster_tile_height / args.unit_num_per_side), int(raster_tile_width / args.unit_num_per_side)
    rgb_unit_height, rgb_unit_width = \
        int(rgb_tile_height / args.unit_num_per_side), int(rgb_tile_width / args.unit_num_per_side)

    # record and dump each unit sample on the fly for memory safety
    train_npz_save_dir = join(args.tgt_data_path, 'train')
    os.makedirs(train_npz_save_dir, exist_ok=True)
    tile_args_list = []
    # loop each row in the tile grid
    for row_idx in range(2):
        # loop each column in the tile grid
        for col_idx in range(7):
            # skip the test tiles
            if row_idx == 1 and col_idx in [2, 3]:
                continue

            # get the current tile of HSI image
            hsi_tile_h_start, hsi_tile_h_end = row_idx * hsi_tile_height, (row_idx + 1) * hsi_tile_height
            hsi_tile_w_start, hsi_tile_w_end = col_idx * hsi_tile_width, (col_idx + 1) * hsi_tile_width
            train_hsi_image_tile = train_hsi_image[
                hsi_tile_h_start: hsi_tile_h_end, hsi_tile_w_start: hsi_tile_w_end
            ]

            # get the current tile of PC image
            raster_tile_h_start, raster_tile_h_end = row_idx * raster_tile_height, (row_idx + 1) * raster_tile_height
            raster_tile_w_start, raster_tile_w_end = col_idx * raster_tile_width, (col_idx + 1) * raster_tile_width
            train_xyz3c_image_tile = train_xyz3c_image[
                raster_tile_h_start: raster_tile_h_end, raster_tile_w_start: raster_tile_w_end
            ]
            train_pc_sem_mask_tile = train_pc_sem_mask[
                raster_tile_h_start: raster_tile_h_end, raster_tile_w_start: raster_tile_w_end
            ]

            # get the current tile of RGB image
            rgb_tile_h_start, rgb_tile_h_end = row_idx * rgb_tile_height, (row_idx + 1) * rgb_tile_height
            rgb_tile_w_start, rgb_tile_w_end = col_idx * rgb_tile_width, (col_idx + 1) * rgb_tile_width
            train_rgb_image_tile = train_rgb_image[
                rgb_tile_h_start: rgb_tile_h_end, rgb_tile_w_start: rgb_tile_w_end
            ]

            # loop each rotation angle choice
            for angle in angle_list:
                angle_key = f'{angle:d}'
                # loop each scale choice
                for scale in scale_list:
                    scale_key = f'{scale:.2f}'
                    tile_args_list.append(
                        [
                            row_idx, col_idx, angle, scale,
                            train_rgb_image_tile, train_hsi_image_tile, train_xyz3c_image_tile, train_pc_sem_mask_tile,
                            rgb_rotate_kwargs_dict[angle_key][scale_key],
                            hsi_rotate_kwargs_dict[angle_key][scale_key],
                            raster_rotate_kwargs_dict[angle_key][scale_key],
                            sem_mask_rotate_kwargs_dict[angle_key][scale_key],
                        ]
                    )

    tile_proc_func = partial(
        scale_rotate_and_clip,
        npz_save_dir=train_npz_save_dir, max_placeholder_ratio=args.max_placeholder_ratio,
        rgb_unit_shape=(rgb_unit_height, rgb_unit_width),
        hsi_unit_shape=(hsi_unit_height, hsi_unit_width),
        raster_unit_shape=(raster_unit_height, raster_unit_width)
    )
    tile_proc_results = [tile_proc_func(tile_args_list)]
    train_metadata = {}
    for item in tile_proc_results:
        train_metadata.update(item)
    return train_metadata


def metadata_generate(args):
    metadata_dict = {}

    # directory init
    rgb_dir = join(args.src_data_path, 'Phase2', 'Final RGB HR Imagery')
    raster_dir = join(args.src_data_path, 'Phase2', 'Lidar GeoTiff Rasters')
    pc_dir = join(args.src_data_path, 'Phase2', 'Lidar Point Cloud Tiles')
    hsi_dir = join(args.src_data_path, 'Phase2', 'FullHSIDataset')
    gt_dir = join(args.src_data_path, 'Phase2', 'TrainingGT')
    vis_save_dir = join(args.tgt_data_path, 'Visualization')
    os.makedirs(vis_save_dir, exist_ok=True)

    # -------------------- HSI Preparation Start -------------------- #
    hsi_ref = envi.open(
        join(hsi_dir, '20170218_UH_CASI_S4_NAD83.hdr'), join(hsi_dir, '20170218_UH_CASI_S4_NAD83.pix')
    )
    # shape (1202, 4172, 50)
    hsi_image = np.array(hsi_ref.load())
    hsi_tile_height, hsi_tile_width = int(hsi_image.shape[0] / 2), int(hsi_image.shape[1] / 7)

    # three default bands are used for visualizing each HSI tile
    default_bands_idx = [int(band) - 1 for band in hsi_ref.metadata['default bands']]
    # shape (1202, 4172, 3)
    hsi_image_vis = hsi_image[:, :, default_bands_idx]
    visualize_nonrgb_image(hsi_image_vis, vis_save_dir=vis_save_dir, vis_file_name='total_hsi_default_bands')
    # -------------------- HSI Preparation End -------------------- #

    # -------------------- Raster Preparation Start -------------------- #
    # shape (2404, 8344, 3), raster data init
    raster_image, (raster_tile_height, raster_tile_width) = prepare_raster_data(raster_dir)
    visualize_nonrgb_image(raster_image, vis_save_dir=vis_save_dir, vis_file_name='total_raster_image')
    # (2404, 8344, 3) & (2404, 8344)
    proj_xyz_image, pc_sem_mask = prepare_pc_data(
        pc_dir, raster_dir, raster_image.shape[0], raster_image.shape[1]
    )
    # (2404, 8344, 6)
    proj_xyz3c_image = np.concatenate([proj_xyz_image, raster_image], axis=-1)

    # visualize projected semantic mask
    color_mapping_all, color_mapping_building = get_color_mapping_for_pc_mask()
    # vis all classes
    pc_sem_mask_rgb_all = color_mapping_all[pc_sem_mask]
    Image.fromarray(
        pc_sem_mask_rgb_all.astype(np.uint8)
    ).save(join(vis_save_dir, f'proj_sem_mask_all.png'))
    # only vis building-related classes (6 & 17)
    pc_sem_mask_rgb_building = color_mapping_building[pc_sem_mask]
    Image.fromarray(
        pc_sem_mask_rgb_building.astype(np.uint8)
    ).save(join(vis_save_dir, f'proj_sem_mask_building.png'))
    # -------------------- Raster Preparation End -------------------- #

    # -------------------- Data Dumping Start -------------------- #
    metadata_dict['train'] = train_split_dump(
        hsi_image=hsi_image, hsi_tile_shape=(hsi_tile_height, hsi_tile_width),
        xyz3c_image=proj_xyz3c_image, pc_sem_mask=pc_sem_mask,
        raster_tile_shape=(raster_tile_height, raster_tile_width),
        rgb_dir=rgb_dir, vis_save_dir=vis_save_dir, args=args
    )

    # rgb:(12020, 23840, 3), gt: (12020, 23840), [old gt: (1202, 2384)], prepare RGB image and GT mask
    test_rgb_image, (rgb_tile_height, rgb_tile_width), gt_mask, (gt_tile_height, gt_tile_width) = \
        prepare_test_data(rgb_dir, gt_dir, vis_save_dir)

    # shape (601, 1192, 50)
    test_hsi_image = hsi_image[hsi_tile_height:, hsi_tile_width * 2: hsi_tile_width * 4]
    visualize_nonrgb_image(test_hsi_image[:, :, default_bands_idx], vis_save_dir=vis_save_dir,
                           vis_file_name='test_hsi_default_bands')

    # shape (1202, 2384, 6)
    test_xyz3c_image = proj_xyz3c_image[raster_tile_height:, raster_tile_width * 2: raster_tile_width * 4]
    visualize_nonrgb_image(test_xyz3c_image[:, :, 3:], vis_save_dir=vis_save_dir, vis_file_name='test_raster_image')

    # shape (1202, 2384)
    test_pc_sem_mask = pc_sem_mask[raster_tile_height:, raster_tile_width * 2: raster_tile_width * 4]
    # vis all classes
    test_pc_sem_mask_rgb_all = color_mapping_all[test_pc_sem_mask]
    Image.fromarray(
        test_pc_sem_mask_rgb_all.astype(np.uint8)
    ).save(join(vis_save_dir, f'test_sem_mask_all.png'))
    # only vis building-related classes (6 & 17)
    test_pc_sem_mask_rgb_building = color_mapping_building[test_pc_sem_mask]
    Image.fromarray(
        test_pc_sem_mask_rgb_building.astype(np.uint8)
    ).save(join(vis_save_dir, f'test_sem_mask_building.png'))

    metadata_dict['test'] = test_split_dump(
        test_rgb_image=test_rgb_image, rgb_tile_shape=(rgb_tile_height, rgb_tile_width),
        test_hsi_image=test_hsi_image, hsi_tile_shape=(hsi_tile_height, hsi_tile_width),
        gt_mask=gt_mask, gt_tile_shape=(gt_tile_height, gt_tile_width),
        test_xyz3c_image=test_xyz3c_image, test_pc_sem_mask=test_pc_sem_mask,
        raster_tile_shape=(raster_tile_height, raster_tile_width),
        args=args
    )
    # -------------------- Data Dumping Start -------------------- #
    # save your generated metadata of each subset into a .json file for reference
    for subset in metadata_dict.keys():
        subset_json_path = join(args.tgt_data_path, f'{subset}.json')
        with open(subset_json_path, 'w') as f:
            json.dump(metadata_dict[subset], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_path', default=f"{DATA_ROOT}/dfc18", type=str)
    parser.add_argument('--tgt_data_path', default=f"{DATA_ROOT}/dfc18_dump", type=str)
    parser.add_argument('--unit_num_per_tile', default=9, type=int)
    parser.add_argument('--max_placeholder_ratio', default=0.5, type=float)
    parser.add_argument('--min_angle', default=0, type=int)
    parser.add_argument('--max_angle', default=160, type=int)
    parser.add_argument('--angle_stride', default=20, type=int)
    parser.add_argument('--min_scale', default=0.8, type=float)
    parser.add_argument('--max_scale', default=1.2, type=float)
    parser.add_argument('--scale_stride', default=0.1, type=float)
    args = parser.parse_args()

    args.unit_num_per_side = math.sqrt(args.unit_num_per_tile)
    if args.unit_num_per_side % 1 > 0:
        raise ValueError
    metadata_generate(args)
