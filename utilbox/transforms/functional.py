import math
import random
from typing import Tuple, List, Union

import numpy as np


def get_random_crop_from_image(image: np.ndarray, scale: Union[List, Tuple], ratio: Union[List, Tuple]):
    height, width = image.shape[:2]
    area = height * width
    aspect_ratio = width / height # ratio default to be the aspect ratio of the original image

    for _ in range(10):
        # randomly pick up scale and ratio candidates
        crop_factor = random.uniform(*scale)
        crop_area = crop_factor * area
        if ratio is not None:
            # select random ratio from log-scale rather than original scale
            # ref to transform classes with the same name in Albumentations or TorchVision
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

        crop_height = int(round(math.sqrt(crop_area / aspect_ratio)))
        crop_width = int(round(math.sqrt(crop_area * aspect_ratio)))
        if 0 < crop_height <= height and 0 < crop_width <= width:
            height_crop_start = random.randint(0, height - crop_height)
            width_crop_start = random.randint(0, width - crop_width)
            return (crop_height, crop_width), (height_crop_start, width_crop_start)

    # Return original image
    return (height, width), (0, 0)
