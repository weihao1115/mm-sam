import torch.nn
import torch.nn as nn

from utilbox.train_utils import fix_params


class BaseImgEncoderWrapper(nn.Module):

    def __init__(self, ori_img_encoder: nn.Module, fix: bool = True):
        super(BaseImgEncoderWrapper, self).__init__()
        self.sam_img_encoder = ori_img_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def set_patch_embed(self, new_patch_embed: torch.nn.Module):
        self.sam_img_encoder.patch_embed = new_patch_embed

    @property
    def patch_embed(self):
        """
        Dynamically get the current patch_embed in the built-in encoder.
        """
        return self.sam_img_encoder.patch_embed

    @property
    def patch_embedding_forward(self):
        assert hasattr(self.sam_img_encoder, "patch_embedding_forward"), \
            "patch_embedding_forward is not implemented for the built-in encoder!"
        return self.sam_img_encoder.patch_embedding_forward

    @property
    def pos_embed(self):
        return self.sam_img_encoder.pos_embed

    @property
    def blocks(self):
        """
        Dynamically get the current blocks in the built-in encoder
        """
        return self.sam_img_encoder.blocks

    @property
    def neck(self):
        return self.sam_img_encoder.neck

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x
