from mm_sam.train_agents.mm_fusion.base import MMFusionSAM


class DFC23MMFusionSAM(MMFusionSAM):
    def __init__(self, train_data: str = "dfc23", **kwargs):
        super(DFC23MMFusionSAM, self).__init__(
            train_data=train_data, x_data_field='sar_images', x_channel_num=1, **kwargs
        )
