import copy
from typing import Optional, Dict, Union, List

import torch
import torch.nn.functional as F
from torch.nn import Module

from mm_sam.models.image_encoders.lora import SAMLoraImgEncoder
from segment_anything.modeling.image_encoder import PatchEmbed
from utilbox.train_utils import data_instance_norm


class XLoraEncoder(Module):
    def __init__(
            self,
            x_data_field: str,
            x_channel_num: int,
            rgb_encoder: Module,
            lora_rank: int = 4,
            norm_type: Optional[str] = 'mean-std',
    ):
        super().__init__()

        self.norm_type = norm_type
        if self.norm_type is not None:
            assert self.norm_type in ['min-max', 'mean-std'], (
                f"Invalid norm_type: {norm_type}! norm_type must be either 'min-max' or 'mean-std'"
            )
        self.x_data_field = x_data_field

        assert rgb_encoder is not None, "rgb_encoder must be provided if align_encoder is not vit!"
        rgb_encoder_copy = copy.deepcopy(rgb_encoder)  # for params safety
        rgb_encoder_conv_proj = rgb_encoder_copy.patch_embed.proj
        x_patch_embed = PatchEmbed(
            kernel_size=rgb_encoder_conv_proj.kernel_size, stride=rgb_encoder_conv_proj.stride,
            in_chans=x_channel_num, embed_dim=rgb_encoder_conv_proj.out_channels
        )
        rgb_encoder_copy.set_patch_embed(x_patch_embed)
        self.encoder = SAMLoraImgEncoder(ori_img_encoder=rgb_encoder_copy, fix=False, rank=lora_rank)

    def preprocess(self, img=None, data_dict: Dict = None):
        if img is None:
            img = data_dict[self.x_data_field]
        x_images, _ = data_instance_norm(img, self.norm_type)
        return x_images

    def image_reshape(self, x_images: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # protect the original data
        x_images = copy.deepcopy(x_images)
        # reshape the images into 1024 x 1024 for SAM compatibility
        if isinstance(x_images, torch.Tensor):
            if (x_images.shape[-2], x_images.shape[-1]) != (1024, 1024):
                x_images = F.interpolate(x_images, (1024, 1024), mode="bilinear", align_corners=False)
        elif isinstance(x_images, List):
            for i in range(len(x_images)):
                if (x_images[i].shape[-2], x_images[i].shape[-1]) != (1024, 1024):
                    x_images[i] = F.interpolate(
                        x_images[i].unsqueeze(0), (1024, 1024), mode="bilinear", align_corners=False
                    )
                else:
                    x_images[i] = x_images[i].unsqueeze(0)
            # (B, x_channel_num, 1024, 1024)
            x_images = torch.cat(x_images, dim=0)
        else:
            raise RuntimeError(
                f"Invalid x_images type: {type(x_images)}. Must be either torch.Tensor or List[torch.Tensor]!"
            )
        x_images = x_images.to(self.device)
        return x_images

    def forward(self, x_images: Union[List[torch.Tensor], torch.Tensor]):
        x_images = self.image_reshape(x_images)
        return self.encoder(x_images)

    @property
    def device(self):
        return self.encoder.patch_embed.proj.weight.device
