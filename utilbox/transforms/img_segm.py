import copy
import random
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict

import torch
import torchvision.transforms.functional as TV_F
from utilbox.transforms.functional import get_random_crop_from_image


class BaseImgSegmTransform(ABC):
    def __init__(self, p: float = 1.0, **kwargs):
        assert 0.0 < p <= 1.0
        self.p = p
        self.init_hook(**kwargs)

    def init_hook(self, **kwargs):
        pass

    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        if random.random() < self.p:
            return self.apply(image, **kwargs)
        else:
            return dict(image=image, **kwargs)

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[BaseImgSegmTransform]):
        if isinstance(transforms, BaseImgSegmTransform):
            transforms = [transforms]
        self.transforms = transforms

    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        data_dict = dict(image=image, **kwargs)
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict


class VerticalFlip(BaseImgSegmTransform):
    def apply(self, image: np.ndarray, **kwargs):
        ret_dict = dict(image=np.ascontiguousarray(image[::-1, ...]))
        for key, value in kwargs.items():
            ret_dict[key] = np.ascontiguousarray(value[::-1, ...])
        return ret_dict


class HorizontalFlip(BaseImgSegmTransform):
    def apply(self, image: np.ndarray, **kwargs):
        ret_dict = dict(image=np.ascontiguousarray(image[:, ::-1, ...]))
        for key, value in kwargs.items():
            ret_dict[key] = np.ascontiguousarray(value[:, ::-1, ...])
        return ret_dict


class RandomRotate90(BaseImgSegmTransform):
    def init_hook(self, rotate_factors: Union[List[int], int] = [1, 3]):
        if isinstance(rotate_factors, int):
            rotate_factors = [rotate_factors, rotate_factors]
        assert rotate_factors[0] >= 1 and rotate_factors[-1] <= 3
        self.rotate_factors = rotate_factors

    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        factor = random.randint(*self.rotate_factors)
        ret_dict = dict(image=np.ascontiguousarray(np.rot90(image, factor)))
        for key, value in kwargs.items():
            ret_dict[key] = np.ascontiguousarray(np.rot90(value, factor))
        return ret_dict


class Transpose(BaseImgSegmTransform):
    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        ret_dict = dict(image=image.transpose(1, 0, 2))
        for key, value in kwargs.items():
            if len(value.shape) == 3:
                ret_dict[key] = value.transpose(1, 0, 2)
            elif len(value.shape) == 2:
                ret_dict[key] = value.transpose(1, 0)
            else:
                raise RuntimeError(f'Unexpected shape {value.shape}')
        return ret_dict


class RandomCrop(BaseImgSegmTransform):
    def init_hook(
            self,
            scale: Union[List[float], float] = [0.1, 1.0],
            ratio: Union[List[float], float] = None
    ):
        if isinstance(scale, float):
            scale = [scale, scale]
        # scale[1] could be larger than 1 to change the random expectation
        assert scale[0] > 0.0 and scale[1] > 0.0
        self.scale = scale

        # default to disable the ratio argument
        if ratio is None:
            self.ratio = ratio
        else:
            if isinstance(ratio, float):
                ratio = [ratio, ratio]
            assert ratio[0] > 0.0 and ratio[1] > 0.0
            self.ratio = ratio

    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        (crop_height, crop_width), (height_crop_start, width_crop_start) = \
            get_random_crop_from_image(image=image, scale=self.scale, ratio=self.ratio)
        height_crop_end = height_crop_start + crop_height
        width_crop_end = width_crop_start + crop_width

        ret_dict = dict()
        for key, value in kwargs.items():
            # get the corresponding crop in mask (may have different shape)
            value_height_crop_start = height_crop_start
            value_width_crop_start = width_crop_start
            value_crop_height = crop_height
            value_crop_width = crop_width
            if image.shape[0] != value.shape[0] or image.shape[1] != value.shape[1]:
                v2i_height_ratio = value.shape[0] / image.shape[0]
                v2i_width_ratio = value.shape[1] / image.shape[1]

                value_height_crop_start = int(value_height_crop_start * v2i_height_ratio)
                value_width_crop_start = int(value_width_crop_start * v2i_width_ratio)
                value_crop_height = int(value_crop_height * v2i_height_ratio)
                value_crop_width = int(value_crop_width * v2i_width_ratio)

            value_height_crop_end = value_height_crop_start + value_crop_height
            value_width_crop_end = value_width_crop_start + value_crop_width
            ret_dict[key] = value[
                value_height_crop_start:value_height_crop_end, value_width_crop_start:value_width_crop_end
            ]

        # slice operation must be done at the end to ensure the precise cropping of image and others
        ret_dict['image'] = image[height_crop_start:height_crop_end, width_crop_start:width_crop_end]
        return ret_dict


class Resize(BaseImgSegmTransform):
    def init_hook(self, resized_shape: Union[Tuple[int], int], resize_func: str = "cv2", skip_mask: bool = False):
        if isinstance(resized_shape, int):
            resized_shape = (resized_shape, resized_shape)
        self.resized_shape = resized_shape

        assert resize_func in ["torch", "cv2"], f"invalid resize_func: {resize_func}! Must be one of `torch`, `cv2`."
        self.resize_func = resize_func
        self.skip_mask = skip_mask

    def resize(self, data_mat: np.ndarray, data_key: str) -> np.ndarray:
        if self.resize_func == "cv2":
            unsqueeze_flag = False
            if len(data_mat.shape) == 3 and data_mat.shape[-1] == 1:
                unsqueeze_flag = True
                data_mat = data_mat[:, :, 0]
            data_mat = cv2.resize(
                data_mat, self.resized_shape,
                # project mask by the nearest interpolation algorithm
                interpolation=cv2.INTER_NEAREST_EXACT if 'mask' in data_key else cv2.INTER_LINEAR
            )
            if unsqueeze_flag:
                data_mat = np.expand_dims(data_mat, axis=-1)
        elif self.resize_func == "torch":
            data_mat = torch.nn.functional.interpolate(
                # (H, W, C) ndarray -> (1, C, H, W) tensor
                torch.from_numpy(data_mat).permute(2, 0, 1).unsqueeze(0), (1024, 1024),
                mode="nearest-exact" if 'mask' in data_key else "bilinear",
                align_corners=None if 'mask' in data_key else False
            ).squeeze(0)
            data_mat = data_mat.permute(1, 2, 0).numpy()
        else:
            raise RuntimeError(f'Unexpected resize_func: {self.resize_func}')

        return data_mat

    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        ret_dict = dict(image=self.resize(image, "image"))
        for key, value in kwargs.items():
            if 'mask' in key and self.skip_mask:
                ret_dict[key] = value
            else:
                ret_dict[key] = self.resize(value, key)
        return ret_dict


class RandomResizedCrop(RandomCrop):
    def init_hook(
            self,
            scale: Union[List[float], float] = [0.1, 1.0],
            ratio: Union[List[float], float] = None,
            resized_shape: Union[Tuple[int], int] = None,
    ):
        super(RandomResizedCrop, self).init_hook(scale, ratio)
        if resized_shape is not None and isinstance(resized_shape, int):
            resized_shape = (resized_shape, resized_shape)
        self.resized_shape = resized_shape

    def apply(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        ori_height, ori_width = image.shape[:2]
        super_dict = super(RandomResizedCrop, self).apply(image=image, **kwargs)
        image = super_dict.pop('image')

        if self.resized_shape is not None:
            resized_shape = (self.resized_shape[1], self.resized_shape[0])
        else:
            resized_shape = (ori_width, ori_height)

        ret_dict = dict(image=cv2.resize(image, resized_shape, interpolation=cv2.INTER_LINEAR))
        for key, value in super_dict.items():
            ret_dict[key] = cv2.resize(
                value, resized_shape,
                # project mask by the nearest interpolation algorithm
                interpolation=cv2.INTER_NEAREST_EXACT if 'mask' in key else cv2.INTER_LINEAR
            )
        return ret_dict


class RandomBrightness(BaseImgSegmTransform):
    def init_hook(self, limit: Union[List[float], float] = 0.1):
        if isinstance(limit, float):
            limit = [-limit, limit]

        assert limit[0] <= 0.0 and limit[1] > 0.0
        self.limit = limit

    def apply(self, image: np.ndarray, **kwargs):
        beta = random.uniform(self.limit[0], self.limit[1])
        image = image + beta * 255
        image = np.clip(image, a_min=0, a_max=255)

        # only apply the brightness transform to image
        return dict(image=image, **kwargs)


class RandomContrast(BaseImgSegmTransform):
    def init_hook(self, limit: Union[List[float], float] = 0.1):
        if isinstance(limit, float):
            limit = [-limit, limit]

        assert limit[0] <= 0.0 and limit[1] > 0.0
        self.limit = limit

    def apply(self, image: np.ndarray, **kwargs):
        alpha = 1.0 + random.uniform(self.limit[0], self.limit[1])
        image_mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = (image - image_mean) * alpha + image_mean
        image = np.clip(image, a_min=0, a_max=255)

        # only apply the contrast transform to image
        return dict(image=image, **kwargs)
