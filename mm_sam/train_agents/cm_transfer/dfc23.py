from mm_sam.train_agents.cm_transfer.base import CMTransferSAM


class DFC23CMTransferSAM(CMTransferSAM):
    def __init__(self, train_data: str = "dfc23", **kwargs):
        super(DFC23CMTransferSAM, self).__init__(
            train_data=train_data, x_data_field='sar_images', x_channel_num=1, **kwargs
        )
