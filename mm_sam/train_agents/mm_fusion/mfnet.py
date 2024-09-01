from mm_sam.datasets import DATASETS
from mm_sam.datasets.transforms import TRANSFORMS_DICT
from mm_sam.train_agents.mm_fusion.base import MMFusionSAM


class MFNetMMFusionSAM(MMFusionSAM):
    def __init__(self, train_data: str = "mfnet", **kwargs):
        super(MFNetMMFusionSAM, self).__init__(
            train_data=train_data, x_data_field='thermal_images', x_channel_num=1, **kwargs
        )

    def build_datasets(
            self,
            train_data: str, train_transforms: str,
            valid_data: str, valid_transforms: str,
            test_data: str, test_transforms: str,
    ):
        if train_transforms is not None:
            train_transforms = TRANSFORMS_DICT[train_transforms]
        self.train_dataset = DATASETS["MMFusion"].build(
            train_data, kwargs=dict(transforms=train_transforms, is_train=True)
        )
        if valid_transforms is not None:
            valid_transforms = TRANSFORMS_DICT[valid_transforms]
        self.valid_dataset = {
            "total": DATASETS["MMFusion"].build(
                valid_data, kwargs=dict(transforms=valid_transforms, is_train=False)
            ),
            "day": DATASETS["MMFusion"].build(
                valid_data, kwargs=dict(transforms=valid_transforms, is_train=False, subset='day')
            ),
            "night": DATASETS["MMFusion"].build(
                valid_data, kwargs=dict(transforms=valid_transforms, is_train=False, subset='night')
            ),
        }
        if test_transforms is not None:
            test_transforms = TRANSFORMS_DICT[test_transforms]
        self.test_dataset = {
            "total": DATASETS["MMFusion"].build(
                test_data, kwargs=dict(transforms=test_transforms, is_train=False)
            ),
            "day": DATASETS["MMFusion"].build(
                test_data, kwargs=dict(transforms=test_transforms, is_train=False, subset='day')
            ),
            "night": DATASETS["MMFusion"].build(
                test_data, kwargs=dict(transforms=test_transforms, is_train=False, subset='night')
            ),
        }
