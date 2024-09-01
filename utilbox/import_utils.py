import functools
import importlib
import inspect
from typing import Dict, Callable, Union


@functools.lru_cache(maxsize=None)
def import_class(class_string) -> Callable:
    class_string = class_string.split('.')
    module_name = '.'.join(class_string[:-1]).strip()
    class_name = class_string[-1].strip()
    return getattr(importlib.import_module(module_name), class_name)


def init_object_from_config(
        config_dict: Dict[Union[str, Callable], Union[Dict, None]],
        extra_args: Dict = None,
        prefix: str = None
):
    assert len(config_dict) == 1, \
        "YAML config for instantiation must contain just one key-value item where the key indicates the target class " \
        "address and the value contains all the arguments for initialization!"

    class_address = list(config_dict.keys())[0]
    if isinstance(class_address, str) and prefix is not None:
        if prefix.endswith('.'):
            prefix = ''.join(prefix[:-1])
        class_address = '.'.join([prefix, class_address])

    init_class = class_address if isinstance(class_address, Callable) else import_class(class_address)
    init_args = {}
    if config_dict[class_address] is not None:
        init_args.update(config_dict[class_address])
    if extra_args is not None:
        init_args.update(extra_args)

    return init_class(**init_args)
