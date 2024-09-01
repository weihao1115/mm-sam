from typing import Dict
from utilbox.import_utils import import_class


def init_transforms_by_config(transform_config: Dict[str, Dict], tgt_package: str, default_args: Dict[str, Dict] = None):
    transform_list = []
    for key, value in transform_config.items():
        transform_args = {}
        if default_args is not None and key in default_args.keys():
            transform_args = default_args[key]
        if value is not None:
            transform_args.update(value)
        transform_list.append(import_class(f'{tgt_package}.{key}')(**transform_args))

    compose_fn = import_class(f'{tgt_package}.Compose')
    return compose_fn(transform_list)
