# DDB-DETR

**Density-Guided Frequency Modulation and Detail–Background Balancing for Tiny Object Detection**

[Overview](#overview) · [Installation](#installation) · [Datasets](#datasets) · [Configuration](#configuration) · [Training](#training) · [Evaluation](#evaluation) · [Inference](#inference)

## Overview

DDB-DETR builds on RT-DETRv2 with a PResNet-34 backbone to distinguish subtle tiny-object details from complex background textures in aerial imagery. Three modules are applied before the hybrid encoder:

| Module | Purpose |
| :--- | :--- |
| **DCEM** | Preserves local details and aggregates contextual information. |
| **DGFQ** | Uses learned density responses to modulate spatial high-pass residuals without additional density annotations. |
| **BSR** | Recalibrates features through foreground enhancement and subtraction of a weighted background branch. |

Each of the three feature levels independently follows **DCEM → DGFQ → BSR**, while retaining the end-to-end prediction interface.

## Installation

### Recommended environment

| Component | Recommendation |
| :--- | :--- |
| Operating system | Ubuntu 20.04 or 22.04 |
| GPU | At least one NVIDIA GPU; 12 GB or more VRAM recommended |
| NVIDIA driver | A driver compatible with the selected CUDA runtime; 525.60.13 or newer for CUDA 12.1 on Linux |
| CUDA runtime | 11.8 or 12.1 |
| Python | 3.10 recommended |
| PyTorch / torchvision | Use a matched pair below |

| PyTorch | torchvision |
| :--- | :--- |
| 2.0 | 0.15 |
| 2.1 | 0.16 |
| 2.2 | 0.17 |
| 2.4 | 0.19 |

GPU memory requirements depend on resolution, batch size, and model size. Python 3.8–3.10 may be used where supported by the selected dependency versions. The CUDA version displayed by nvidia-smi indicates driver capability; a local CUDA toolkit does not need to match the runtime bundled with prebuilt PyTorch wheels.

### 1. Create a Conda environment

~~~bash
conda create -n rtdetrv2 python=3.10 -y
conda activate rtdetrv2
~~~

### 2. Install PyTorch

Choose **one** combination compatible with your NVIDIA driver.

**CUDA 11.8:**

~~~bash
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
~~~

**CUDA 12.1:**

~~~bash
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
~~~

Verify GPU access:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
~~~

The availability check must print **True** before GPU training.

### 3. Download the source

For **DDB-DETR**, clone this repository:

~~~bash
git clone https://github.com/shaonian276-web/DDB-DETR.git
cd DDB-DETR/rtdetrv2_pytorch
~~~

For the original RT-DETRv2 implementation, clone the upstream repository separately:

~~~bash
git clone https://github.com/lyuwenyu/RT-DETR.git
cd RT-DETR/rtdetrv2_pytorch
~~~

Run all remaining commands from the selected repository's **rtdetrv2_pytorch** directory.

### 4. Install dependencies

~~~bash
python -m pip install -r requirements.txt
~~~

If needed, use the Tsinghua PyPI mirror for this step:

~~~bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~

Optional visualization and analysis utilities:

~~~bash
python -m pip install "numpy==1.26.4" "opencv-python<4.12" matplotlib tqdm pandas seaborn
python -m pip check
~~~

The NumPy pin is intended for the Python 3.10 environment above. The OpenCV constraint avoids releases requiring NumPy 2. Keep the selected torch/torchvision pair consistent after installation.

The dependency set may include torch, torchvision, faster-coco-eval, PyYAML, tensorboard, scipy, pycocotools, NumPy, OpenCV, matplotlib, tqdm, and pandas. ONNX export and deployment may additionally require onnx and onnxruntime-gpu; choose versions compatible with your deployment environment. Refer to **requirements.txt** for project-specific requirements.

### 5. Check the directory structure

~~~bash
ls
~~~

The RT-DETRv2 layout typically includes:

~~~text
rtdetrv2_pytorch/
├── configs/
├── src/
├── tools/
├── references/
├── requirements.txt
└── README.md
~~~

The training entry point is **tools/train.py**.

## Datasets

### Official downloads

| Dataset | Official resources |
| :--- | :--- |
| **VisDrone2019-DET** | [Training, validation, and test downloads (Baidu Netdisk / Google Drive)](https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images) |
| **AI-TOD-v2** | [Download (Google Drive)](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing) · [Project and documentation](https://github.com/Chasel-Tsui/mmdet-aitod) |

Use the updated **AI-TOD-v2 annotations** and follow each dataset's usage and citation requirements.

### COCO format

RT-DETRv2 uses COCO-style detection annotations. A convenient layout is:

~~~text
dataset/
├── train2017/
│   ├── image_001.jpg
│   └── ...
├── val2017/
│   ├── image_002.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
~~~

Convert YOLO annotations to COCO JSON before training. Ensure filenames, image IDs, category IDs, and bounding boxes match the images. COCO boxes use [x, y, width, height] in pixels.

## Configuration

The examples below illustrate the RT-DETRv2 workflow with the lightweight R18 configuration:

~~~text
configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml
~~~

**The R18 example is not the PResNet-34 DDB-DETR experiment configuration.** For DDB-DETR experiments, select the configuration for the intended backbone and modules. Use the same model configuration for training, evaluation, and inference, and confirm the example file exists in your checkout.

Set the number of classes in the effective configuration, for example:

~~~yaml
num_classes: 10  # Replace with your dataset's class count.
~~~

Follow the top-level configuration's **__include__** entries to locate the dataset and dataloader settings. Image folders and annotation paths may be defined in an included dataset configuration rather than directly in **configs/rtdetrv2/include/dataloader.yml**. Update both training and validation paths.

For custom datasets, review **remap_mscoco_category** and disable COCO-specific category mapping when it does not match your labels. Also review batch size, worker count, output directory, and training schedule. Reduce batch size if GPU memory is insufficient; for distributed training, ensure the total batch size is divisible by the GPU count.

## Training

### Single GPU

~~~bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  --use-amp \
  --seed=0
~~~

### Multiple GPUs

Example using four GPUs:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  --use-amp \
  --seed=0
~~~

Choose an available master port when running concurrent jobs.

### Background training

~~~bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  --use-amp \
  --seed=0 > train.log 2>&1 &
~~~

Monitor the log:

~~~bash
tail -f train.log
~~~

### Fine-tune pretrained weights

Use **-t** to initialize a new training run from compatible pretrained weights:

~~~bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -t path/to/pretrained.pth \
  --use-amp \
  --seed=0
~~~

Replace the checkpoint placeholder with an actual file. Inspect the loading log for missing or mismatched parameters when changing the backbone, modules, or class count. Fine-tuning (**-t**) and resuming an interrupted run (**-r**) serve different purposes.

## Evaluation

Evaluate a checkpoint using its matching model configuration and validation annotations:

~~~bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -r path/to/checkpoint.pth \
  --test-only
~~~

## Inference

For checkouts containing **references/deploy/rtdetrv2_torch.py**, run single-image inference with:

~~~bash
python references/deploy/rtdetrv2_torch.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -r path/to/checkpoint.pth \
  --im-file path/to/image.jpg \
  --device cuda:0
~~~

Confirm the supported arguments in your checkout:

~~~bash
python references/deploy/rtdetrv2_torch.py --help
~~~

Replace all example paths and use the configuration that matches your checkpoint.

## Acknowledgments

This project builds on [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR). We thank the original authors and the VisDrone and AI-TOD-v2 dataset maintainers for their work.
