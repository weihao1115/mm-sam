from mm_sam.train_agents.mm_fusion.base import MMFusionSAM


class DFC18HSIMMFusionSAM(MMFusionSAM):
    def __init__(self, train_data: str = "dfc18", **kwargs):
        super(DFC18HSIMMFusionSAM, self).__init__(
            train_data=train_data, x_data_field='hsi_images', x_channel_num=48, **kwargs
        )


class DFC18PCMMFusionSAM(MMFusionSAM):
    def __init__(self, train_data: str = "dfc18", **kwargs):
        super(DFC18PCMMFusionSAM, self).__init__(
            train_data=train_data, x_data_field='proj_xyz3c_images', x_channel_num=6, **kwargs
        )
