from mm_sam.train_agents.cm_transfer.base import CMTransferSAM


class DFC18HSICMTransferSAM(CMTransferSAM):
    def __init__(self, train_data: str = "dfc18_hsi", **kwargs):
        super(DFC18HSICMTransferSAM, self).__init__(
            train_data=train_data, x_data_field='hsi_images', x_channel_num=48, **kwargs
        )


class DFC18PCCMTransferSAM(CMTransferSAM):
    def __init__(self, train_data: str = "dfc18_pc", **kwargs):
        super(DFC18PCCMTransferSAM, self).__init__(
            train_data=train_data, x_data_field='proj_xyz3c_images', x_channel_num=6, **kwargs
        )
