import importlib
import pkgutil
from os.path import dirname, join, abspath
from typing import Any, Dict, Callable


class Registry:
    """Global registry dictionary registry"""
    def __init__(self):
        self._registry = {}
        self._import_flag = False

    def regiter_all_modules(self, package_name: str):
        if not self._import_flag:
            # iteratively import all the submodule to enable automatic class registration
            package = importlib.import_module(package_name)
            package_dir = dirname(abspath(package.__file__))
            self._import_and_register(package_name, package_dir)
            self._import_flag = True

    def _import_and_register(self, package_name: str, package_dir: str):
        for _, module_name, ispkg in pkgutil.iter_modules([package_dir]):
            full_module_name = f"{package_name}.{module_name}"
            if ispkg:
                sub_package_dir = join(package_dir, module_name)
                self._import_and_register(full_module_name, sub_package_dir)
            else:
                importlib.import_module(full_module_name)

    def register(self, key: str):
        def decorator(value: Callable):
            assert isinstance(value, Callable), "Your registered value must be callable!"
            if key not in self._registry:
                self._registry[key] = value
            elif self._registry[key] != value:
                raise RuntimeError(f"Key {key} has been registered twice with different values! "
                                   f"{self._registry[key]} v.s. {value}")
            return value
        return decorator

    def build(self, key: str, kwargs: Dict[str, Any] = None):
        if kwargs is None: kwargs = {}

        if key in self._registry:
            return self._registry[key](**kwargs)
        else:
            raise KeyError(
                f'{key} not registered! Available keys: {list(self._registry.keys())}'
            )

    def __contains__(self, key):
        return key in self._registry

    @property
    def available_keys(self):
        return list(self._registry.keys())
