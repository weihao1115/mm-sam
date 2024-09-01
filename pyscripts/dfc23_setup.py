import argparse
import json
import os
from typing import List, Dict, Union

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from os.path import join, exists
from tqdm import tqdm

from utilbox.global_config import DATA_ROOT


def segm_to_mask(segm: Union[List[int], Dict[str, Union[List[int], str]]], height: int, width: int):
    """Adapted from pycocotools.
    https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py#L282
    https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py#L301
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        rle = segm
    mask = maskUtils.decode(rle)
    return mask


def main(src_dir: str):
    train_dir = join(src_dir, 'train')
    assert exists(train_dir), f'Train folder is not found in {src_dir}!'

    json_path = join(src_dir, 'roof_fine_train.json')
    assert exists(json_path), f'Annotation json file is not found: {json_path}!'
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    corr_json_path = join(src_dir, 'roof_fine_train_corrected.json')
    assert exists(corr_json_path), f'Corrected annotation json file is not found: {corr_json_path}!'
    with open(corr_json_path, 'r') as f:
        corr_json_dict = json.load(f)

    gt_mask_dir = join(train_dir, 'gt_mask')
    os.makedirs(gt_mask_dir, exist_ok=True)
    object_masks_dir = join(train_dir, 'object_masks')
    os.makedirs(object_masks_dir, exist_ok=True)
    corr_gt_mask_dir = join(train_dir, 'gt_masks_corrected')
    os.makedirs(corr_gt_mask_dir, exist_ok=True)
    corr_object_masks_dir = join(train_dir, 'object_masks_corrected')
    os.makedirs(corr_object_masks_dir, exist_ok=True)

    metadata_dict = {}
    for image in tqdm(json_dict['images']):
        file_name, height, width, image_id = image.values()
        image_name = file_name.replace('.tif', '')

        segm_list = [ann['segmentation'] for ann in json_dict['annotations'] if ann['image_id'] == image_id]
        mask_list = [segm_to_mask(segm, height, width) for segm in segm_list]
        if len(mask_list) == 0:
            mask_list.append(np.zeros((height, width)))
        object_masks = np.stack(mask_list, axis=0, dtype=np.float32)
        gt_mask = np.max(object_masks, axis=0) * 255.

        gt_mask_path = join(gt_mask_dir, f'{image_name}.png')
        object_masks_path = join(object_masks_dir, f'{image_name}.npz')
        np.savez_compressed(object_masks_path, object_masks=object_masks)
        Image.fromarray(gt_mask.astype(np.uint8)).save(gt_mask_path)

        corr_segm_list = [ann['segmentation'] for ann in corr_json_dict['annotations'] if ann['image_id'] == image_id]
        corr_mask_list = [segm_to_mask(corr_segm, height, width) for corr_segm in corr_segm_list]
        if len(corr_mask_list) == 0:
            corr_mask_list.append(np.zeros((height, width)))
        corr_object_masks = np.stack(corr_mask_list, axis=0, dtype=np.float32)
        corr_gt_mask = np.max(corr_object_masks, axis=0) * 255.

        corr_gt_mask_path = join(corr_gt_mask_dir, f'{image_name}.png')
        corr_object_masks_path = join(corr_object_masks_dir, f'{image_name}.npz')
        np.savez_compressed(corr_object_masks_path, object_masks=corr_object_masks)
        Image.fromarray(corr_gt_mask.astype(np.uint8)).save(corr_gt_mask_path)

        rgb_file_path = join(train_dir, 'rgb', file_name)
        assert exists(rgb_file_path), f'RGB file is not found: {rgb_file_path}!'
        sar_file_path = join(train_dir, 'sar', file_name)
        assert exists(sar_file_path), f'SAR file is not found: {sar_file_path}!'

        metadata_dict[image_name] = dict(
            # save the relative path to src_dir
            rgb_file_path=rgb_file_path.replace(src_dir + '/', ''),
            sar_file_path=sar_file_path.replace(src_dir + '/', ''),
            gt_mask_path=gt_mask_path.replace(src_dir + '/', ''),
            object_masks_path=object_masks_path.replace(src_dir + '/', '') + '::object_masks',
            corr_gt_mask_path=corr_gt_mask_path.replace(src_dir + '/', ''),
            corr_object_masks_path=corr_object_masks_path.replace(src_dir + '/', '') + '::object_masks',
        )

    metadata_path = join(src_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default=f"{DATA_ROOT}/dfc23")
    args = parser.parse_args()
    main(**vars(args))
