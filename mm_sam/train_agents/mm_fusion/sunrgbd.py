from mm_sam.train_agents.mm_fusion.base import MMFusionSAM


class SunRGBDMMFusionSAM(MMFusionSAM):
    def __init__(self, train_data: str = "sunrgbd", **kwargs):
        super(SunRGBDMMFusionSAM, self).__init__(
            train_data=train_data, x_data_field='depth_images', x_channel_num=1, **kwargs
        )
