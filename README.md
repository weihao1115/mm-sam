<h1 align="center">Segment Anything with Multiple Modalities
</h1>
<p align="center">

<a href="https://arxiv.org/abs/2408.09085" target='_blank'>
    <img src="https://img.shields.io/badge/Arxiv-2408.09085-b31b1b.svg?logo=arXiv">
  </a><h5 align="center">
    <em>
        <a href="https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en">Aoran Xiao*</a>,
        <a href="https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en">Weihao Xuan*</a>,
        <a href="https://scholar.google.co.jp/citations?user=CH-rTXsAAAAJ&hl=en">Heli Qi</a>,
        <a href="https://scholar.google.co.jp/citations?user=uOAYTXoAAAAJ&hl=en">Yun Xing</a>,
        <a href="https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=ja">Naoto Yokoya^</a>,
        <a href="https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en">Shijian Lu^</a> 
       <br>(* indicates co-first authors with equal contributions. ^ indicates the corresponding authors.)
    </em>
</h5><p align="center">


## About
Robust and accurate segmentation of scenes has become one core functionality in various visual recognition and navigation tasks. This has inspired the recent development of Segment Anything Model (SAM), a foundation model for general mask segmentation. However, SAM is largely tailored for single-modal RGB images, limiting its applicability to multi-modal data captured with widely-adopted sensor suites, such as LiDAR plus RGB, depth plus RGB, thermal plus RGB, etc. We develop MM-SAM, an extension and expansion of SAM that supports cross-modal and multi-modal processing for robust and enhanced segmentation with different sensor suites. MM-SAM features two key designs, namely, unsupervised cross-modal transfer and weakly-supervised multi-modal fusion, enabling label-efficient and parameter-efficient adaptation toward various sensor modalities. It addresses three main challenges: 1) adaptation toward diverse non-RGB sensors for single-modal processing, 2) synergistic processing of multi-modal data via sensor fusion, and 3) mask-free training for different downstream tasks. Extensive experiments show that MM-SAM consistently outperforms SAM by large margins, demonstrating its effectiveness and robustness across various sensors and data modalities.

## News
- **(2024/9)** We released the testing code. Thank you for your waiting!

## Method
![overall_pipeline](./figs/mm_sam.png "overall_pipeline")

## Outline
- [Installation](#installation)
  - [Path Registration](#path-registration)
  - [SAM Pretrained Models](#sam-pretrained-models)
- [Data](#data)
  - [SUNRGBD](#sunrgbd)
  - [MFNet](#mfnet)
  - [Data Fusion Contest 2018 (DFC18)](#data-fusion-contest-2018-dfc18)
  - [Data Fusion Contest 2023 (DFC23)](#data-fusion-contest-2023-dfc23)
- [Train (Coming soon)](#train)
- [Test](#test)
  - [Model Library](#model-library)
- [Inference](#inference)
  - [PyTorch-alone](#pytorch-alone)
    - [UCMT Models](#ucmt-models)
    - [WMMF Models](#wmmf-models)
  - [HuggingFace Integration (Coming soon)](#huggingface-integration)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [Related Projects](#related-projects)

## Installation

Please clone our project to your local machine and prepare our environment by the following commands:

```
conda create -n mm_sam python=3.10 -y
conda activate mm_sam
cd /your/path/to/destination/mm-sam
(mm_sam) python -m pip install -e .
(mm_sam) conda install -c conda-forge gdal==3.8.3
```

**Note:** if your local CUDA version is 11.8, you may need to manually install `torch` and `torchvision` by
`pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118`
before `python -m pip install -e .`

The code has been tested on A100/A6000/V100 with Python 3.10, CUDA 11.8/12.1 and Pytorch 2.1.2.
Any other devices and environments may require to update the code for compatibility.

### Path Registration
Please register the following 3 paths in `utilbox/global_config.py`:
- `EXP_ROOT`: the root directory where the checkpoint files of trained models will be saved.
- `DATA_ROOT`: the root directory where you want to place the datasets.
- `PRETRAINED_ROOT`: the root directory where you want to place the SAM pretrained model files.

### SAM Pretrained Models
Please download SAM pretrained models from [this link](https://github.com/facebookresearch/segment-anything#model-checkpoints).
You should have the following file structure:
```
{your PRETRAINED_ROOT}
    |___sam_vit_b_01ec64.pth
    |___sam_vit_l_0b3195.pth
    |___sam_vit_h_4b8939.pth
```

## Data
Please follow the instructions below to prepare each dataset.

### SUNRGBD

```
conda activate mm_sam
(mm_sam): python -m pyscripts.sunrgbd_setup
```

You are expected to have the following file structure:
```
{your DATA_ROOT}
   |___sunrgbd
       |___SUNRGBD
       |   |___kv1
       |   |___kv2
       |   |___realsense
       |   |___xtion
       |___SUNRGBDtoolbox
       |___test_depth.txt
       |___test_label.txt
       |___test_rgb.txt
       |___train_depth.txt
       |___train_label.txt
       |___train_rgb.txt
```

### MFNet
Please download `ir_seg_dataset` from [this link](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw) to `{your DATA_ROOT}`
```
cd {your DATA_ROOT}
unzip ir_seg_dataset.zip
mv ir_seg_dataset MFNet
rm ir_seg_dataset.zip
```
Then, open `{your DATA_ROOT}/MFNet/make_flip.py` and change `root_dir` to `{your DATA_ROOT}/MFNet` and run
```
cd MFNet
conda activate mm_sam
(mm_sam): python make_flip.py
```
You are expected to have the following file structure:
```
{your DATA_ROOT}
   |___MFNet
       |___images
       |   |___00001D.png
       |   |___00001D_flip.png
       |   |___...
       |   |___01606D.png
       |___labels
       |   |___00001D.png
       |   |___00001D_flip.png
       |   |___...
       |   |___01606D.png
       |___train.txt
       |___val.txt
       |___test.txt
       |___test_day.txt
       |___test_night.txt
```

### Data Fusion Contest 2018 (DFC18)
```
cd {your DATA_ROOT}
wget http://hyperspectral.ee.uh.edu/QZ23es1aMPH/2018IEEE/phase2.zip
unzip phase2.zip
mv 2018IEEE_Contest/ dfc18
rm phase2.zip
```
Then, download `test_gt_mask_ori_scale.png` from [this link](https://drive.google.com/drive/folders/1G5cWuAB2TGSXs8gE9_93ZABY0nmlGuGc) to `{your DATA_ROOT}/dfc18/Phase2/TrainingGT`.
You are expected to have the following file structure:
```
{your DATA_ROOT}
    |___dfc18
        |___Phase2
            |___Final RGB HR Imagery
            |___FullHSIDataset
            |___Lidar GeoTiff Rasters
            |___Lidar Point Cloud Tiles
            |___TrainingGT
                |___...
                |___test_gt_mask_ori_scale.png
```
Finally, run
```
conda activate mm_sam
(mm_sam) python -m pyscripts.dfc18_setup
```
`dfc18_dump` will be created in `{your DATA_ROOT}` with the following file structure:
```
{your DATA_ROOT}
    |___dfc18_dump
        |___test
        |   |___row0_col0.npz
        |   |___...
        |   |___row2_col5.npz
        |___train
        |   |___trow0_tcol0_angle0_scale0.80_urow0_ucol0.npz
        |   |___...
        |   |___trow1_tcol6_angle160_scale1.20_urow3_ucol3.npz
        |___Visualization
        |___test.json
        |___train.json
```
**Note:** `dfc18_dump` will consume around 170G disk space.

### Data Fusion Contest 2023 (DFC23)
Download `train.tgz` from [this link](https://drive.google.com/uc?id=19TCMnmUN_CH7YxUjZfsXhKshzcG7p7Zq) to `{your DATA_ROOT}/dfc23` and run
```
cd {your DATA_ROOT}/dfc23
tar -xzvf train.tgz
rm train.tgz
```
You are expected to have the following file structure:
```
{your DATA_ROOT}
    |___dfc23
        |___train
        |   |___rgb
        |   |___sar
        |___roof_fine_train.json
        |___roof_fine_train_corrected.json
```
**Note:** If you are decompressing `train.tgz` on a Linux OS, you may need to run 
```
cd {your DATA_ROOT}/dfc23
find ./ -name ".*.tif" -type f -delete
find ./ -name ".*.json" -type f -delete
```
to remove useless files introduced by the OS mismatch.

Then, run
```
conda activate mm_sam
(mm_sam): python -m pyscripts.dfc23_setup
```
Finally, the file structure will be
```
{your DATA_ROOT}
    |___dfc23
        |___train
        |   |___rgb
        |   |___sar
        |___roof_fine_train.json
        |___roof_fine_train_corrected.json
        |___metadata.json
```

## Train
Coming soon.

## Test
Test will be automatically done after each training job and the results will be printed to the console.
If you want to merely evaluate your trained checkpoint, you can run the following command after you finish the training:
```
conda activate mm_sam
(mm_sam): python -m pyscripts.launch --config_name {cm_transfer or mm_fusion}/{dataset}_{gpu_config} --test_only True
```

### Model Library
We also provide our checkpoints used in our paper. You can find two kinds of checkpoints at [our HuggingFace page](https://huggingface.co/weihao1115/mmsam_ckpt/tree/main):
- `{dataset}_{modality}_encoder_vit_b.pth`: The X encoder of `{modality}` data trained by UCMT.
- `{dataset}_{modality}_sfg_vit_b.pth`: The SFG module of `{modality}` data trained by WMMF based on `{dataset}_{modality}_encoder_vit_b.pth`.

You can download them to your local machine and evaluate their performance by our configuration files using the command:
- For UCMT experiments, 
  ```
  conda activate mm_sam
  (mm_sam): python -m pyscripts.launch --config_name cm_transfer/{dataset}_{gpu_config} --test_only True --ckpt_path /your/path/to/{dataset}_{modality}_encoder_vit_b.pth
  ```
- For WMMF experiments, first modify `agent_kwargs[x_encoder_ckpt_path]` in your target configuration `./config/mm_fusion/{dataset}_{gpu_config}.yaml` to `/your/path/to/{dataset}_{modality}_encoder_vit_b.pth`.
  Then, run
  ```
  conda activate mm_sam
  (mm_sam): python -m pyscripts.launch --config_name mm_fusion/{dataset}_{gpu_config} --test_only True --ckpt_path /your/path/to/{dataset}_{modality}_encoder_vit_b.pth
  ```
**Note:** we recommend you to download the model checkpoints to `{your PRETRAIN_ROOT}/mmsam_ckpt` for easy management.

## Inference
We provide user-friendly API for model inference by either PyTorch-alone APIs and HuggingFace APIs. 
You can download our checkpoint files along with the corresponding configuration files from [our HuggingFace page](https://huggingface.co/weihao1115/mmsam_ckpt/tree/main).

### PyTorch-alone
#### UCMT Models
Please download **BOTH** `{dataset}_{modality}_encoder_vit_b.pth` and `{dataset}_{modality}_encoder_vit_b.yaml` to `{your PRETRAIN_ROOT}/mmsam_ckpt` from [our HuggingFace page](https://huggingface.co/weihao1115/mmsam_ckpt/tree/main).
Below is an example to conduct inference of UCMT models on an image sample from SUNRGBD dataset.
```python
import torch
from mm_sam.models.sam import SAMbyUCMT
from utilbox.global_config import DATA_ROOT
from utilbox.data_load.read_utils import read_depth_from_disk

ucmt_sam = SAMbyUCMT.from_pretrained(x_encoder_ckpt_path="mmsam_ckpt/sunrgbd_depth_encoder_vit_b")
ucmt_sam = ucmt_sam.to("cuda").eval()

# depth_image: (H, W, 1)
depth_image_path = f"{DATA_ROOT}/sunrgbd/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/depth_bfx/0000103.png"
depth_image = read_depth_from_disk(depth_image_path, return_tensor=True)
ucmt_sam.set_infer_img(img=depth_image, channel_last=True)

# 1. Single-box Inference
# box_coords: (4,)
box_coords = torch.Tensor([291, 53, 729, 388])
pred_masks, pred_ious = ucmt_sam.infer(box_coords=box_coords)
# pred_mask: (H, W) 0-1 binary mask in torch.Tensor type
pred_mask = pred_masks[0].squeeze()

# 2. Multi-box Inference
# 2.1. ensemble predicted mask
# box_coords: (2, 4)
box_coords = torch.Tensor([[291, 53, 729, 388], [23, 289, 729, 529]])
pred_masks, pred_ious = ucmt_sam.infer(box_coords=box_coords)
# pred_mask: (H, W) 0-1 binary mask
pred_mask = pred_masks[0].squeeze()

# 2.2. separate predicted mask
pred_masks, pred_ious = ucmt_sam.infer(box_coords=box_coords, return_all_prompt_masks=True)
# pred_mask: (2, H, W) 0-1 binary mask in torch.Tensor type
pred_mask = pred_masks[0].squeeze()
```
If you want to use your trained checkpoint files to conduct inference, please initialize UCMT model by
```python
from mm_sam.models.sam import SAMbyUCMT
from utilbox.global_config import EXP_ROOT

ucmt_sam = SAMbyUCMT.from_pretrained(
  x_encoder_ckpt_path=f"{EXP_ROOT}/cm_transfer/sunrgbd_1x4090/checkpoints/best_mean_nonzero_fore_iu_models/your_checkpoint.pth", 
  x_encoder_config_path="mmsam_ckpt/sunrgbd_depth_encoder_vit_b"
)
```
Above is an example for the UCMT models trained on SUNRGBD. Please change the name to your corresponding dataset.

#### WMMF Models
Please download **ALL** 
- `{dataset}_{modality}_encoder_vit_b.pth`
- `{dataset}_{modality}_encoder_vit_b.yaml`
- `{dataset}_{modality}_sfg_vit_b.pth`
- `{dataset}_{modality}_sfg_vit_b.yaml`

to `{your PRETRAIN_ROOT}/mmsam_ckpt` from [our HuggingFace page](https://huggingface.co/weihao1115/mmsam_ckpt/tree/main).
Below is an example to conduct inference of WMMF models on an image sample from SUNRGBD dataset.
```python
import torch
from mm_sam.models.sam import SAMbyWMMF
from utilbox.global_config import DATA_ROOT
from utilbox.data_load.read_utils import read_image_as_rgb_from_disk, read_depth_from_disk

wmmf_sam = SAMbyWMMF.from_pretrained(
    x_encoder_ckpt_path="mmsam_ckpt/sunrgbd_depth_encoder_vit_b",
    sfg_ckpt_path="mmsam_ckpt/sunrgbd_depth_sfg_vit_b",
)
wmmf_sam = wmmf_sam.to("cuda").eval()

# rgb_image: (H, W, 3) 0-255 RGB image
rgb_image_path = f"{DATA_ROOT}/sunrgbd/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg"
rgb_image = read_image_as_rgb_from_disk(rgb_image_path, return_tensor=True)

# depth_image: (H, W, 1)
depth_image_path = f"{DATA_ROOT}/sunrgbd/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/depth_bfx/0000103.png"
depth_image = read_depth_from_disk(depth_image_path, return_tensor=True)
wmmf_sam.set_infer_img(rgb_img=rgb_image, x_img=depth_image, channel_last=True)

# 1. Single-box Inference
# box_coords: (4,)
box_coords = torch.Tensor([291, 53, 729, 388])
pred_masks, pred_ious = wmmf_sam.infer(box_coords=box_coords)
# pred_mask: (H, W)
pred_mask = pred_masks[0].squeeze()

# 2. Multi-box Inference
# 2.1. ensemble predicted mask
# box_coords: (2, 4)
box_coords = torch.Tensor([[291, 53, 729, 388], [23, 289, 729, 529]])
pred_masks, pred_ious = wmmf_sam.infer(box_coords=box_coords)
# pred_mask: (H, W) 0-1 binary mask
pred_mask = pred_masks[0].squeeze()

# 2.2. separate predicted mask
pred_masks, pred_ious = wmmf_sam.infer(box_coords=box_coords, return_all_prompt_masks=True)
# pred_mask: (2, H, W) 0-1 binary mask
pred_mask = pred_masks[0].squeeze()
```
If you want to use your trained checkpoint files to conduct inference, please initialize WMMF model by
```python
from mm_sam.models.sam import SAMbyWMMF
from utilbox.global_config import EXP_ROOT

wmmf_sam = SAMbyWMMF.from_pretrained(
  x_encoder_ckpt_path=f"{EXP_ROOT}/cm_transfer/sunrgbd_1x4090/checkpoints/best_mean_nonzero_fore_iu_models/your_checkpoint.pth", 
  x_encoder_config_path="mmsam_ckpt/sunrgbd_depth_encoder_vit_b",
  sfg_ckpt_path=f"{EXP_ROOT}/mm_fusion/sunrgbd_1x4090/checkpoints/best_mean_nonzero_fore_iu_models/your_checkpoint.pth", 
  sfg_config_path="mmsam_ckpt/sunrgbd_depth_sfg_vit_b",
)
```
Above is an example for the WMMF models trained on SUNRGBD. Please change the name to your corresponding dataset.

## HuggingFace Integration
Coming soon.

## Citation

If you find this work helpful, please kindly consider citing our paper:

```bibtex
@article{mmsam,
  title={Segment Anything with Multiple Modalities}, 
  author={Aoran Xiao and Weihao Xuan and Heli Qi and Yun Xing and Naoto Yokoya and Shijian Lu},
  journal={arXiv preprint arXiv:2408.09085},
  year={2024}
}
```

## Acknowledgement
We acknowledge the use of the following public resources throughout this work: [Segment Anything Model](https://github.com/facebookresearch/segment-anything), and [LoRA](https://github.com/microsoft/LoRA).

## Related Projects
Find our other projects for visual foundation models!

[CAT-SAM: Conditional Tuning for Few-Shot Adaptation of Segment Anything Model](https://github.com/weihao1115/cat-sam), ECCV 2024, Oral Paper.
