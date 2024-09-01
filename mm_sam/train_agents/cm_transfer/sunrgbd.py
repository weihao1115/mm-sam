from mm_sam.train_agents.cm_transfer.base import CMTransferSAM


class SunRGBDCMTransferSAM(CMTransferSAM):
    def __init__(self, train_data: str = "sunrgbd", **kwargs):
        super(SunRGBDCMTransferSAM, self).__init__(
            train_data=train_data, x_data_field='depth_images', x_channel_num=1, **kwargs
        )
