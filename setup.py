from setuptools import setup, find_packages
from os.path import join, curdir, exists


base_requires_path = join(curdir, "envs", "base_requirements.txt")
if not exists(base_requires_path):
    raise RuntimeError(f"Please run 'python -m pip install -e .' at the project root!")
with open(base_requires_path, 'r') as file:
    install_requires = [line.strip() for line in file if line.strip()]

# extra requires
install_requires += [
    "torch==2.1.2",
    "torchvision==0.16.2",
    "laspy==2.5.3",
    "spectral==0.23.1",
    "huggingface-hub==0.24.6",
    "safetensors==0.4.4",
]

setup(
    name="MM-SAM",
    version="0.1",
    description="The official repository of `Segment Anything with Multiple Modalities`",
    platforms=["linux-64", "osx-64"],
    license="Apache-2.0",
    url="",
    install_requires=install_requires,
    packages=find_packages(),
)
