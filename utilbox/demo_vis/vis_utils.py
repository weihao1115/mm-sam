from os.path import join

import numpy as np
from PIL import Image


def nonrgb_to_rgb(nonrgb_image: np.ndarray, log_vis: bool = False, log_num: int = 1):
    if len(nonrgb_image.shape) == 2:
        nonrgb_image = np.repeat(nonrgb_image[:, :, None], repeats=3, axis=-1)
    elif len(nonrgb_image.shape) == 3 and nonrgb_image.shape[-1] == 1:
        nonrgb_image = np.repeat(nonrgb_image, repeats=3, axis=-1)
    elif len(nonrgb_image.shape) != 3 or nonrgb_image.shape[-1] != 3:
        raise RuntimeError("Non-RGB image should be either (H, W), (H, W, 1) or (H, W, 3).")

    channel_min = np.min(nonrgb_image.reshape(-1, 3), axis=0)
    channel_max = np.max(nonrgb_image.reshape(-1, 3), axis=0)
    rgb_image = (nonrgb_image - channel_min) / (channel_max - channel_min + 1e-10)
    if log_vis:
        for _ in range(log_num):
            rgb_image = np.log((np.e - 1) * rgb_image + 1)
    rgb_image = np.round(rgb_image * 255.)
    return rgb_image


def visualize_nonrgb_image(nonrgb_image: np.ndarray, vis_save_dir: str, vis_file_name: str):
    nonrgb_image_vis = nonrgb_to_rgb(nonrgb_image)
    vis_img = Image.fromarray(nonrgb_image_vis.astype(np.uint8))
    vis_img.save(join(vis_save_dir, f'{vis_file_name}.png'))
    return vis_img
