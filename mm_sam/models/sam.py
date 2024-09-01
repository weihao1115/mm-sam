import copy
from os.path import join, dirname, basename, exists

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from typing import Tuple, List, Union, Dict, Optional
from huggingface_hub import PyTorchModelHubMixin

from mm_sam.models.module_lib.sfg import SelectiveFusionGate
from mm_sam.models.module_lib.x_encoder import XLoraEncoder
from mm_sam.models.image_encoders.base import BaseImgEncoderWrapper
from mm_sam.models.prompt_encoders.base import BasePromptEncodeWrapper
from mm_sam.models.mask_decoders.base import BaseMaskDecoderWrapper

from utilbox.yaml_utils import load_yaml
from utilbox.global_config import PRETRAINED_ROOT
from utilbox.import_utils import import_class


class SAMWrapper(nn.Module):
    sam_ckpt_path = {
        "default": f"{PRETRAINED_ROOT}/sam_vit_h_4b8939.pth",
        "vit_b": f"{PRETRAINED_ROOT}/sam_vit_b_01ec64.pth",
        "vit_l": f"{PRETRAINED_ROOT}/sam_vit_l_0b3195.pth",
        "vit_h": f"{PRETRAINED_ROOT}/sam_vit_h_4b8939.pth",
    }

    def __init__(
            self,
            model_type: str,
            sam_model_registry: Union[str, Dict] = 'segment_anything.sam_model_registry',
            multimask_output: bool = False,
            fix_img_en: bool = True,
            fix_prompt_en: bool = True,
            fix_mask_de: bool = True,
    ):
        super(SAMWrapper, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        sam_ckpt_path = self.sam_ckpt_path[model_type]

        if isinstance(sam_model_registry, str):
            sam_model_registry = import_class(sam_model_registry)
        ori_sam = sam_model_registry[model_type](sam_ckpt_path)
        self.sam_img_size = (ori_sam.image_encoder.img_size, ori_sam.image_encoder.img_size)
        self.register_buffer("pixel_mean", ori_sam.pixel_mean, False)
        self.register_buffer("pixel_std", ori_sam.pixel_std, False)
        self.mask_threshold = ori_sam.mask_threshold
        self.multimask_output = multimask_output

        self.image_encoder = BaseImgEncoderWrapper(
            ori_img_encoder=ori_sam.image_encoder, fix=fix_img_en
        )
        self.prompt_encoder = BasePromptEncodeWrapper(
            ori_prompt_encoder=ori_sam.prompt_encoder, fix=fix_prompt_en
        )
        self.mask_decoder = BaseMaskDecoderWrapper(
            ori_mask_decoder=ori_sam.mask_decoder, fix=fix_mask_de
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
            self,
            imgs: Union[torch.Tensor, None],
            point_coords: List[Union[torch.Tensor, None]],
            point_labels: List[Union[torch.Tensor, None]],
            box_coords: List[Union[torch.Tensor, None]],
            noisy_masks: List[Union[torch.Tensor, None]],
            img_feats: torch.Tensor = None,
            ori_img_size: List[Tuple] = None,
            output_mask_size: List[Tuple] = None,
            multimask_output: bool = None,
            rgb_pixel_norm: bool = True
    ):
        # img_feats has the higher priority than imgs for encoder feature
        if img_feats is not None:
            assert ori_img_size is not None, "ori_img_size must be given if img_feats is provided!"
            x = img_feats
            # preprocess the value scale of points and boxes
            _, point_coords, box_coords = self.preprocess(
                point_coords=point_coords, box_coords=box_coords, ori_img_size=self.ori_infer_img_size
            )
        else:
            ori_img_size = [(imgs[i].shape[-2], imgs[i].shape[-1]) for i in range(len(imgs))]
            imgs, point_coords, box_coords = self.preprocess(
                imgs=imgs, point_coords=point_coords, box_coords=box_coords,
                ori_img_size=ori_img_size, pixel_norm=rgb_pixel_norm
            )
            # imgs here is normalized with the shape of (B, 3, 1024, 1024)
            x = self.image_encoder(imgs)

        points, boxes, masks = self.convert_raw_prompts_to_triple(
            point_coords=point_coords, point_labels=point_labels,
            box_coords=box_coords, noisy_masks=noisy_masks, batch_size=len(x)
        )

        if multimask_output is None:
            multimask_output = self.multimask_output

        # We do prompt encoding and mask decoding for every single image one by one
        outputs = []
        for batch_idx in range(len(x)):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points[batch_idx], boxes=boxes[batch_idx], masks=masks[batch_idx],
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=x[batch_idx].unsqueeze(0),
                prompt_encoder=self.prompt_encoder,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # # Select the correct mask or masks for output
            # if multimask_output:
            #     # mask with the highest score
            #     max_iou_idx = torch.argmax(iou_predictions, dim=1)
            #     low_res_masks = low_res_masks[torch.arange(low_res_masks.size(0)), max_iou_idx].unsqueeze(1)

            # rescale the mask size back to original image size
            ori_res_masks = self.postprocess(
                pred_masks=low_res_masks,
                output_mask_size=ori_img_size[batch_idx] if output_mask_size is None else output_mask_size[batch_idx]
            )
            outputs.append(
                {
                    "low_res_logits": low_res_masks,
                    "ori_res_logits": ori_res_masks,
                    "iou_predictions": iou_predictions
                }
            )
        return outputs

    @staticmethod
    def check_img_type_and_shape(img, channel_last: bool):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif isinstance(img, List):
            for i in range(len(img)):
                if isinstance(img[i], np.ndarray):
                    img[i] = torch.from_numpy(img[i])
        elif not isinstance(img, torch.Tensor):
            raise RuntimeError(
                "`img` must be given as one of the following format:\n"
                "- a list of 2-dim or 3-dim np.ndarray\n"
                "- a list of 2-dim or 3-dim torch.Tensor\n"
                "- a 2-dim or 3-dim or 4-dim np.ndarray\n"
                "- a 2-dim or 3-dim or 4-dim torch.Tensor\n"
            )

        if isinstance(img, torch.Tensor):
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
                img = [img]
            elif len(img.shape) == 3:
                img = img.permute(2, 0, 1) if channel_last else img
                img = [img]
            elif len(img.shape) == 4:
                img = [i.permute(2, 0, 1) if channel_last else i for i in img]
            else:
                raise RuntimeError("If `img` are given as a torch.Tensor, it must be either 2-dim, 3-dim or 4-dim!")
        elif isinstance(img, List):
            for i in range(len(img)):
                if len(img[i].shape) == 2:
                    img[i] = img[i].unsqueeze(0)
                elif len(img[i].shape) == 3:
                    img[i] = img[i].permute(2, 0, 1) if channel_last else img[i]
                else:
                    raise RuntimeError("If `img` are given as a list, each element must be either 2-dim or 3-dim!")

        return img


    def set_infer_img(
            self,
            img: Union[List[np.ndarray | torch.Tensor], torch.Tensor, np.ndarray] = None,
            data_dict: Dict = None,
            channel_last: bool = False,
            pixel_norm: bool = True
    ):
        """

        Args:
            img: `img` must be given as one of the following format
                - a list of 2-dim or 3-dim np.ndarray
                - a list of 2-dim or 3-dim torch.Tensor
                - a 2-dim or 3-dim or 4-dim np.ndarray
                - a 2-dim or 3-dim or 4-dim torch.Tensor
            data_dict: If `img` is not given, `data_dict` must be given where the key `images` must exist with the
                value in the same format as `img`.
            channel_last: If images are given as 3-dim torch.Tensor or np.ndarray, `channel_last` indicates whether the
                image is in the shape of (H, W, C) with `channel_last=True` or (C, H, W) with `channel_last=False`
            pixel_norm: whether to normalize the images by SAM's built-in pixel_mean and pixel_std.

        """
        assert (img is not None) or (data_dict is not None), "One of `img` or `data_dict` must be provided!"
        if img is None:
            img = data_dict['images']
        img = self.check_img_type_and_shape(img, channel_last)

        img_return = copy.deepcopy(img)
        for i in range(len(img_return)):
            img_return[i] = img_return[i].to(device=self.device, dtype=torch.float32)
        self.ori_infer_img_size = [(i.shape[-2], i.shape[-1]) for i in img_return]

        # register encoding features for the input inference image
        img_return, _, _ = self.preprocess(
            imgs=img_return, ori_img_size=self.ori_infer_img_size, pixel_norm=pixel_norm
        )
        self.img_features = self.image_encoder(img_return)

    def infer(
            self,
            point_coords: Union[torch.Tensor, List[torch.Tensor | None], None] = None,
            box_coords: Union[torch.Tensor, List[torch.Tensor | None], None] = None,
            output_mask_size: Optional[List[Tuple]] = None,
            return_all_prompt_masks: bool = False,
            return_logits: bool = False,
            return_low_res: bool = False,
            multimask_output: Optional[bool] = None,
    ):
        if not hasattr(self, 'img_features') or not hasattr(self, 'ori_infer_img_size'):
            raise RuntimeError(
                "Image encoder features have not been registered! Please call set_infer_img() before infer().")
        point_coords, point_labels, box_coords, noisy_masks = (
            self.proc_raw_prompts(point_coords=point_coords, box_coords=box_coords)
        )

        # Mask prediction by prompt points
        outputs = self.forward(
            imgs=None, img_feats=self.img_features,
            ori_img_size=self.ori_infer_img_size, output_mask_size=output_mask_size,
            point_coords=point_coords, point_labels=point_labels, box_coords=box_coords, noisy_masks=noisy_masks,
            multimask_output=multimask_output
        )

        masks = [item['low_res_logits' if return_low_res else 'ori_res_logits'] for item in outputs]
        if not return_logits:
            # discretize masks into 0-1 matrix
            masks = [self.discretize_mask(m) for m in masks]

        if not return_all_prompt_masks:
            assert not return_logits, "masks must be discretized if you want to assemble all prompt masks of an image."
            # sum up the prediced masks by all the prompts of a single image
            masks = [torch.clamp(torch.sum(m, dim=0, keepdim=True), max=1.0) for m in masks]

        return masks, [item['iou_predictions'] for item in outputs]

    def convert_raw_prompts_to_triple(
            self,
            point_coords: List[Optional[torch.Tensor]],
            point_labels: List[Optional[torch.Tensor]],
            box_coords: List[Optional[torch.Tensor]],
            noisy_masks: List[Optional[torch.Tensor]],
            batch_size: int
    ):
        points, boxes, masks = [], [], []
        for batch_idx in range(batch_size):
            points_idx = None
            if point_coords[batch_idx] is not None:
                # prompt point coordinates must in the shape of (N_prompt, N_point, 2)
                point_coords_idx = point_coords[batch_idx]
                if len(point_coords_idx.shape) == 2:
                    # if the first dim (N_prompt) is not given, default to generate just one output mask
                    point_coords_idx = point_coords_idx.unsqueeze(0)
                if len(point_coords_idx.shape) != 3:
                    raise RuntimeError(
                        "Each prompt point coordinate in the list must be in the shape of (N_prompt, N_point, 2) "
                        "where N_prompt is the number of output masks and N_point is the number of prompt points!")
                if point_coords_idx.size(-1) != 2:
                    raise RuntimeError("Each prompt point must be given as a two-element vector!")

                point_labels_idx = point_labels[batch_idx]
                if len(point_labels_idx.shape) == 1:
                    # if the first dim (N_prompt) is not given, default to generate just one output mask
                    point_labels_idx = point_labels_idx.unsqueeze(0)
                if len(point_labels_idx.shape) != 2:
                    raise RuntimeError(
                        "Each prompt point coordinate in the list must be in the shape of (N_prompt, N_point, 2) "
                        "where N_prompt is the number of output masks and N_point is the number of prompt points!")

                points_idx = (
                    point_coords_idx.to(dtype=torch.float32, device=self.device),
                    point_labels_idx.to(dtype=torch.float32, device=self.device)
                )
            points.append(points_idx)

            boxes_idx = None
            if box_coords[batch_idx] is not None:
                # prompt box coordinates must in the shape of (N_prompt, 2)
                box_coords_idx = box_coords[batch_idx]
                if len(box_coords_idx.shape) == 1:
                    # if the first dim (N_prompt) is not given, default to generate just one output mask
                    box_coords_idx = box_coords_idx.unsqueeze(0)
                if len(box_coords_idx.shape) != 2:
                    raise RuntimeError(
                        "Each prompt point coordinate in the list must be in the shape of (N_prompt, 4) "
                        "where N_prompt is the number of output masks!")
                if box_coords_idx.size(-1) != 4:
                    raise RuntimeError("Each prompt box must be given as a four-element vector!")

                boxes_idx = box_coords_idx.to(dtype=torch.float32, device=self.device)
            boxes.append(boxes_idx)

            masks_idx = None
            if noisy_masks[batch_idx] is not None:
                noisy_masks_idx = noisy_masks[batch_idx]
                if len(noisy_masks_idx.shape) == 2:
                    # if the first dim (N_prompt) is not given, default to generate just one output mask
                    noisy_masks_idx = noisy_masks_idx[None, None, :, :]
                if len(noisy_masks_idx.shape) == 3:
                    noisy_masks_idx = noisy_masks_idx[None, :, :]
                if len(noisy_masks_idx.shape) != 4:
                    raise RuntimeError(
                        "Each prompt mask in the list must be in the shape of (N, 1, 256, 256) "
                        "where N is the number of output masks!")
                if noisy_masks_idx.size(1) != 1:
                    raise RuntimeError("Please only give one prompt mask for each output prompt!")
                if noisy_masks_idx.size(-2) != 256 and noisy_masks_idx.size(-1) != 256:
                    raise RuntimeError("Each prompt mask must have width and height of 256!")

                masks_idx = noisy_masks_idx.to(dtype=torch.float32, device=self.device)
            masks.append(masks_idx)

        return points, boxes, masks

    def proc_raw_prompts(
            self,
            point_coords: List[Union[List, torch.Tensor, None]] = None,
            box_coords: List[Union[List, torch.Tensor, None]] = None,
            point_labels=None,
            noisy_masks=None,
    ):
        if not hasattr(self, 'ori_infer_img_size'):
            raise RuntimeError(
                "Image encoder features have not been registered! Please call set_infer_img() before infer()."
            )

        if point_coords is not None:
            if isinstance(point_coords, torch.Tensor):
                if len(point_coords.shape) == 1:
                    point_coords = [point_coords[None, None, :]]
                elif len(point_coords.shape) == 2:
                    point_coords = [point_coords.unsqueeze(0)]
                elif len(point_coords.shape) == 3:
                    point_coords = [point_coords]
                elif len(point_coords.shape) == 4:
                    point_coords = [p_c for p_c in point_coords]
                else:
                    raise RuntimeError(
                        "If point_coords is given as a torch.Tensor, it must be either "
                        "4-dim (B, N_prompt, N_point, 2), 3-dim (N_prompt, N_point, 2), 2-dim (N_point, 2) or 1-dim (2,)!"
                    )
            elif isinstance(point_coords, List):
                for i in range(len(point_coords)):
                    if point_coords[i] is None:
                        continue

                    if not isinstance(point_coords[i], torch.Tensor):
                        raise RuntimeError(
                            "If point_coords is given as a list, each element must be either "
                            "a 3-dim (N_prompt, N_point, 2), 2-dim (N_point, 2) or 1-dim (2,) torch.Tensor!"
                        )
                    if len(point_coords[i].shape) == 1:
                        point_coords[i] = point_coords[i][None, None, :]
                    elif len(point_coords[i].shape) == 2:
                        point_coords[i] = point_coords[i].unsqueeze(0)
                    elif len(point_coords[i].shape) != 3:
                        raise RuntimeError(
                            "If point_coords is given as a list, each element must be either "
                            "a 3-dim (N_prompt, N_point, 2), 2-dim (N_point, 2) or 1-dim (2,) torch.Tensor!"
                        )
            else:
                raise RuntimeError(
                    "`point_coords` must be given as one of the following format:\n"
                    "- a list of 1-dim (2,), 2-dim (N_point, 2) or 3-dim (N_prompt, N_point, 2) torch.Tensor\n"
                    "- a 1-dim (2,), 2-dim (N_point, 2), 3-dim (N_prompt, N_point, 2) or 4-dim (B, N_prompt, N_point, 2) torch.Tensor\n"
                )
            for p_c in point_coords:
                if p_c is None:
                    continue
                if p_c.shape[-1] != 2:
                    raise RuntimeError("each prompt point must be given as a two-element coordinate of [X, Y]!")
                if len(p_c.shape) != 3:
                    raise RuntimeError

        if box_coords is not None:
            if isinstance(box_coords, torch.Tensor):
                if len(box_coords.shape) == 1:
                    box_coords = [box_coords.unsqueeze(0)]
                elif len(box_coords.shape) == 2:
                    box_coords = [box_coords]
                elif len(box_coords.shape) == 3:
                    box_coords = [b_c for b_c in box_coords]
                else:
                    raise RuntimeError(
                        "If box_coords is given as a torch.Tensor, "
                        "it must be either 3-dim (B, N_prompt, 4), 2-dim (N_prompt, 4), or 1-dim (4,)!"
                    )
            elif isinstance(box_coords, List):
                for i in range(len(box_coords)):
                    if box_coords[i] is None:
                        continue

                    if not isinstance(box_coords[i], torch.Tensor):
                        raise RuntimeError(
                            "If box_coords is given as a list, each element must be either "
                            "a 1-dim (4,), or 2-dim (N_prompt, 4) torch.Tensor!"
                        )
                    if len(box_coords[i].shape) == 1:
                        box_coords[i] = box_coords[i].unsqueeze(0)
                    elif len(box_coords[i].shape) != 2:
                        raise RuntimeError(
                            "If box_coords is given as a list, each element must be either "
                            "a 2-dim (N_prompt, 4), or 1-dim (4,) torch.Tensor!"
                        )
            else:
                raise RuntimeError(
                    "`box_coords` must be given as one of the following format:\n"
                    "- a list of 1-dim (4,) or 2-dim (N_prompt, 4) torch.Tensor\n"
                    "- a 1-dim (4,) or 2-dim (N_prompt, 4) or 3-dim (B, N_prompt, 4) torch.Tensor\n"
                )
            for b_c in box_coords:
                if b_c is None:
                    continue
                if b_c.shape[-1] != 4:
                    raise RuntimeError(
                        "each prompt box must be given as a four-element coordinate of "
                        "[X_{top_left}, Y_{top_left}, X_{bottom_right}, Y_{bottom_right}]!"
                    )
                if len(b_c.shape) != 2:
                    raise RuntimeError

        if noisy_masks is not None:
            raise NotImplementedError("Inference by noisy masks has not been supported yet~")

        def proc_coords(input_coords):
            if input_coords is not None:
                _tmp_input_coords = []
                for i_c in input_coords:
                    if isinstance(i_c, torch.Tensor):
                        _tmp_input_coords.append(i_c)
                    elif i_c is None or len(i_c[0]) == 0:
                        _tmp_input_coords.append(None)
                    elif len(i_c[0]) > 0:
                        _tmp_input_coords.append(torch.FloatTensor(i_c))
                    else:
                        raise RuntimeError(
                            "The element of the input coords must be one of List, torch.Tensor, and None. "
                            f"Got {type(i_c)}!!"
                        )
                input_coords = _tmp_input_coords
            return input_coords

        # produce prompt points and their corresponding labels
        point_coords, point_labels = proc_coords(point_coords), proc_coords(point_labels)
        if point_coords is None:
            point_coords = [None for _ in self.ori_infer_img_size]
            point_labels = [None for _ in self.ori_infer_img_size]
        elif point_labels is None:
            point_labels = [
                torch.ones((p_c.size(0), p_c.size(1)), dtype=torch.long, device=p_c.device) if p_c is not None else None
                for p_c in point_coords
            ]
        if len(point_coords) != len(point_labels):
            raise RuntimeError("point_coords and point_labels must have the same length!")

        # produce prompt boxes
        box_coords = proc_coords(box_coords)
        if box_coords is None:
            box_coords = [None for _ in self.ori_infer_img_size]

        # We disable prompt masks for prediction
        if noisy_masks is None:
            noisy_masks = [None for _ in self.ori_infer_img_size]

        return point_coords, point_labels, box_coords, noisy_masks

    def discretize_mask(self, masks_logits):
        return torch.gt(masks_logits, self.mask_threshold).float()

    def preprocess(
            self,
            ori_img_size: List[Tuple] = None,
            imgs: List[torch.Tensor] = None,
            point_coords: List[Union[torch.Tensor, None]] = None,
            box_coords: List[Union[torch.Tensor, None]] = None,
            interpolate_mode: str = "bilinear",
            pixel_norm: bool = True
    ):
        assert imgs is not None or point_coords is not None or box_coords is not None, \
            "At least one of imgs, point_coords, and box_coords must be given!!"
        assert imgs is not None or ori_img_size is not None, \
            "At least one of imgs and ori_img_size must be given!!"

        if ori_img_size is None:
            ori_img_size = [
                (imgs[i].shape[-2], imgs[i].shape[-1]) for i in range(len(imgs))
            ]

        # rescale the data in the out-place manner
        imgs_return, point_coords_return, box_coords_return = \
            copy.deepcopy(imgs), copy.deepcopy(point_coords), copy.deepcopy(box_coords)
        # resize each image into a 4-dim tensor for interpolate
        if imgs_return is not None:
            for i in range(len(imgs_return)):
                if len(imgs_return[i].shape) == 3:
                    imgs_return[i] = imgs_return[i].unsqueeze(0)
                if len(imgs_return[i].shape) != 4:
                    raise RuntimeError(
                        f'Wrong image shape! Each image in your given list must be (C, H, W), '
                        f'but got {imgs_return[i].shape}!'
                    )

        # loop each image
        for i in range(len(ori_img_size)):
            # skip the one with the same size as SAM input
            if ori_img_size[i] == self.sam_img_size:
                if pixel_norm and imgs_return is not None:
                    imgs_return[i] = (imgs_return[i] - self.pixel_mean) / self.pixel_std
                continue

            if imgs_return is not None:
                # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
                # change the mode from bilinear to nearest
                imgs_return[i] = F.interpolate(
                    imgs_return[i], self.sam_img_size, mode=interpolate_mode,
                    align_corners=None if interpolate_mode in ("nearest", "area", "nearest-exact") else False
                )
                if pixel_norm:
                    imgs_return[i] = (imgs_return[i] - self.pixel_mean) / self.pixel_std

            h_scale = self.sam_img_size[0] / ori_img_size[i][0]
            w_scale = self.sam_img_size[1] / ori_img_size[i][1]
            if point_coords_return is not None and point_coords_return[i] is not None:
                # ensure that coords are float numbers before scaling
                if point_coords_return[i].dtype != torch.float32:
                    point_coords_return[i] = point_coords_return[i].to(dtype=torch.float32)
                # scale the x-axis by the width scaler
                point_coords_return[i][:, :, 0] *= w_scale
                # scale the y-axis by the height scaler
                point_coords_return[i][:, :, 1] *= h_scale
                # make sure that the point coordinates are in the form of integers
                point_coords_return[i] = torch.round(point_coords_return[i])
            if box_coords_return is not None and box_coords_return[i] is not None:
                # ensure that coords are float numbers before scaling
                if box_coords_return[i].dtype != torch.float32:
                    box_coords_return[i] = box_coords_return[i].to(dtype=torch.float32)
                # scale the x-axis by the width scaler
                box_coords_return[i][:, 0] *= w_scale
                box_coords_return[i][:, 2] *= w_scale
                # scale the y-axis by the height scaler
                box_coords_return[i][:, 1] *= h_scale
                box_coords_return[i][:, 3] *= h_scale
                # make sure that the point coordinates are in the form of integers
                box_coords_return[i] = torch.round(box_coords_return[i])

        # organize all the image tensors into a larger matrix
        if imgs_return is not None:
            imgs_return = torch.cat(imgs_return, dim=0)
        return imgs_return, point_coords_return, box_coords_return

    @staticmethod
    def postprocess(
            pred_masks: torch.Tensor,
            output_mask_size: Tuple,
            interpolate_mode: str = "bilinear"
    ):
        # rescale the mask size back to original image size
        pred_mask_size = (pred_masks.size(-2), pred_masks.size(-1))
        if pred_mask_size != output_mask_size:
            if len(pred_masks.shape) == 3:
                pred_masks = pred_masks.unsqueeze(1)
            # 'bilinear' mode will produce non-deterministic gradients during training.
            pred_masks = F.interpolate(
                pred_masks, output_mask_size, mode=interpolate_mode,
                align_corners=None if interpolate_mode in ("nearest", "area", "nearest-exact") else False
            )
        return pred_masks


class SAMbyUCMT(
    SAMWrapper, PyTorchModelHubMixin,
    repo_url="https://github.com/weihao1115/mm-sam", pipeline_tag="mask-generation", license="mit",
):
    def __init__(
            self,
            model_type: str,
            x_data_field: str,
            x_channel_num: int,
            x_norm_type: Optional[str],
            x_lora_rank: int = 4,
            x_encoder_ckpt_path: Optional[str] = None
    ):
        super().__init__(model_type=model_type)
        self.x_data_field = x_data_field
        x_encoder = XLoraEncoder(
            x_data_field=x_data_field,
            x_channel_num=x_channel_num,
            lora_rank=x_lora_rank,
            norm_type=x_norm_type,
            rgb_encoder=self.image_encoder
        )
        self.image_encoder = x_encoder
        if x_encoder_ckpt_path is not None:
            self.load_x_encoder(x_encoder_ckpt_path)

    def load_x_encoder(self, x_encoder_ckpt_path: str):
        if not x_encoder_ckpt_path.startswith("/"):
            x_encoder_ckpt_path = join(PRETRAINED_ROOT, x_encoder_ckpt_path)
        if not x_encoder_ckpt_path.endswith(".pth"):
            x_encoder_ckpt_path = f"{x_encoder_ckpt_path}.pth"
        x_encoder_ckpt = torch.load(x_encoder_ckpt_path, map_location=self.device)
        self.image_encoder.load_state_dict(x_encoder_ckpt)

    def set_infer_img(
            self,
            img: Union[List[torch.Tensor], torch.Tensor] = None,
            data_dict: Optional[Dict] = None,
            channel_last: bool = False,
            pixel_norm: bool = True
    ):
        if img is None:
            img = data_dict[self.x_data_field]
        img = self.check_img_type_and_shape(img, channel_last)
        x_images = self.image_encoder.preprocess(img=img)
        # SAM RGB pre-norm is disabled for x-modality images
        # channel_last is disabled since x_images is already channel-first
        super().set_infer_img(img=x_images, channel_last=False, pixel_norm=False)


class SAMbyWMMF(
    SAMWrapper, PyTorchModelHubMixin,
    repo_url="https://github.com/weihao1115/mm-sam", pipeline_tag="mask-generation", license="mit",
):
    def __init__(
            self,
            model_type: str,
            x_data_field: str,
            x_channel_num: int,
            x_norm_type: Optional[str],
            x_lora_rank: int = 4,
            x_encoder_ckpt_path: Optional[str] = None,
            sfg_filter_num: int = 1,
            sfg_intermediate_channels: int = 32,
            sfg_filter_type: str = "conv2d",
            sfg_ckpt_path: Optional[str] = None
    ):
        super().__init__(model_type=model_type)
        self.x_data_field = x_data_field
        x_encoder = XLoraEncoder(
            x_data_field=x_data_field,
            x_channel_num=x_channel_num,
            lora_rank=x_lora_rank,
            norm_type=x_norm_type,
            rgb_encoder=self.image_encoder
        )
        self.x_encoder = x_encoder
        if x_encoder_ckpt_path is not None:
            self.load_x_encoder(x_encoder_ckpt_path)

        fusion_module = SelectiveFusionGate(
            filter_num=sfg_filter_num,
            intermediate_channels=sfg_intermediate_channels,
            filter_type=sfg_filter_type
        )
        self.fusion_module = fusion_module
        if sfg_ckpt_path is not None:
            self.load_sfg(sfg_ckpt_path)

    def load_x_encoder(self, x_encoder_ckpt_path: str):
        if not x_encoder_ckpt_path.startswith("/"):
            x_encoder_ckpt_path = join(PRETRAINED_ROOT, x_encoder_ckpt_path)
        if not x_encoder_ckpt_path.endswith(".pth"):
            x_encoder_ckpt_path = f"{x_encoder_ckpt_path}.pth"
        x_encoder_ckpt = torch.load(x_encoder_ckpt_path, map_location=self.device)
        self.x_encoder.load_state_dict(x_encoder_ckpt)

    def load_sfg(self, sfg_ckpt_path: str):
        if not sfg_ckpt_path.startswith("/"):
            sfg_ckpt_path = join(PRETRAINED_ROOT, sfg_ckpt_path)
        if not sfg_ckpt_path.endswith(".pth"):
            sfg_ckpt_path = f"{sfg_ckpt_path}.pth"
        sfg_ckpt = torch.load(sfg_ckpt_path, map_location=self.device)
        self.fusion_module.load_state_dict(sfg_ckpt)

    def set_infer_img(
            self,
            rgb_img: Union[List[np.ndarray | torch.Tensor], torch.Tensor, np.ndarray] = None,
            x_img: Union[List[np.ndarray | torch.Tensor], torch.Tensor, np.ndarray] = None,
            data_dict: Optional[Dict] = None,
            channel_last: bool = False,
            pixel_norm: bool = True
    ):
        if rgb_img is None:
            rgb_img = data_dict["rgb_images"]
        if x_img is None:
            x_img = data_dict[self.x_data_field]

        super().set_infer_img(img=rgb_img, channel_last=channel_last, pixel_norm=pixel_norm)
        rgb_feats = self.img_features

        x_img = self.check_img_type_and_shape(x_img, channel_last)
        x_images = self.x_encoder.preprocess(img=x_img)
        x_feats = self.x_encoder(x_images)

        fuse_feats, fuse_feat_masks = self.fusion_module(feat_list=[rgb_feats, x_feats])
        self.img_features = fuse_feats
