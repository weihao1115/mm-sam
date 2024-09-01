from utilbox.registry_utils import Registry

DATASETS = {
    "CMTransfer": Registry(),
    "MMFusion": Registry(),
}
for v in DATASETS.values():
    v.regiter_all_modules(package_name=__name__)
