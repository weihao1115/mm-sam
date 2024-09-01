"""
This file contains all the general configurations for your project.

Please register the information based on your local machine here before making your project.

"""
from os.path import exists, abspath, dirname

# the root directory of the project source codes
PROJECT_ROOT = abspath(dirname(dirname(__file__)))
# the root directory that places the experiment log data
EXP_ROOT = ''
# the root directory of the local datasets
DATA_ROOT = ''
# the root directory of the pretrained models
PRETRAINED_ROOT = ''


# runtime checking for mandatory configurations
config_dict = dict(
    PROJECT_ROOT=PROJECT_ROOT, EXP_ROOT=EXP_ROOT, DATA_ROOT=DATA_ROOT, PRETRAINED_ROOT=PRETRAINED_ROOT
)
for k, v in config_dict.items():
    if v is None or v == '':
        raise RuntimeError(f"Please register {k} in utilbox.global_config!")
    elif not exists(v):
        raise RuntimeError(f"Your registered {k} ({v}) does not exist! Please create it manually!")
