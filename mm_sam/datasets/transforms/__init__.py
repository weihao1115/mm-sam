TRANSFORMS_DICT = {
    "cmtransfer_v0": {
        "RandomCrop": dict(scale=[0.8, 1.0])
    },
    "cmtransfer_v1": {
        "RandomCrop": dict(scale=[0.8, 1.0]),
        "Resize": dict(resized_shape=1024, skip_mask=True),
    },
    "resize_1024": {
        "Resize": dict(resized_shape=1024, skip_mask=True),
    },
}
