import torch.nn as nn

from utilbox.train_utils import fix_params


class BasePromptEncodeWrapper(nn.Module):

    def __init__(self, ori_prompt_encoder: nn.Module, fix: bool = True):
        super(BasePromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()
