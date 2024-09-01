import torch.nn as nn

from utilbox.train_utils import fix_params


class BaseMaskDecoderWrapper(nn.Module):

    def __init__(self, ori_mask_decoder: nn.Module, fix: bool = True):
        super(BaseMaskDecoderWrapper, self).__init__()
        self.sam_mask_decoder = ori_mask_decoder
        if fix:
            fix_params(self.sam_mask_decoder)

    def forward(self, image_embeddings, prompt_encoder, sparse_embeddings, dense_embeddings, multimask_output=True):
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=prompt_encoder.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        return low_res_masks, iou_predictions

    @property
    def vis_module(self):
        assert hasattr(self.sam_mask_decoder, 'vis_module'), "vis_module is not defined in the built-in mask decoder!"
        return self.sam_mask_decoder.vis_module
