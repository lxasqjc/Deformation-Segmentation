# Deformation for Segmentation in PyTorch

This is a PyTorch implementation of Learning to Downsample for Segmentation of Ultra-High Resolution Images.

### Table of Contents
1. [Environment-Setup](#environment-Setup)
1. [Data-preparation](#data-preparation)
1. [Reproduce](#reproduce)

## Environment-Setup

### Special dependencies for interp2d

```
pip install Cython
cd spatial
python setup.py build_ext --inplace
```

### Install dependencies
Install dependencies with one of the following options:
* Method 1: Pip installation:
```
python -m pip install -r requirements.txt
```
* Method 2: Conda installation with miniconda3 PATH ```/home/miniconda3/```:
```
conda env create -f deform_seg_env.yml
conda activate deform_seg_env
```
This code was tested with python 3.7, pytorch 1.2 and CUDA 11.0

## Data preparation
1. Download the [Cityscapes](https://www.cityscapes-dataset.com/), [DeepGlobe](https://competitions.codalab.org/competitions/18468) and [PCa-histo](to-be-released) datasets.

2. Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── annotations
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   └── images
│       ├── testing
│       ├── training
│       └── validation
├── histomri
│   ├── train
│   │   ├── images
│   │   ├── labels
│   └── val
│   │   ├── images
│   │   ├── labels
├── deepglob
│   ├── land-train
│   └── land_train_gt_processed

note Histo_MRI is the PCa-histo dataset
````

3. Data list .odgt files are provided in ```./data``` prepare correspondingly for local datasets


## Reproduce
full configuration bash provided to reproduced paper results, suitable for large scale experiment in multiple GPU Environment, Syncronized Batch Normalization are deployed.

### Training
Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, last checkpoints by default are saved in folder ```ckpt```.
```bash
python3 train_fove.py --gpus $GPUS --cfg $CFG
```
- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

* Bashes and configurations are provided to reproduce our results:

NOTE: you will need to specify your root path 'SEG_ROOT' for ```DATASET.root_dataset``` option in those scripts.

```bash
bash quick_start_bash/cityscape_64_128_ours.sh
bash quick_start_bash/cityscape_64_128_uniform.sh
bash quick_start_bash/deepglob_300_300_ours.sh
bash quick_start_bash/deepglob_300_300_uniform.sh
bash quick_start_bash/pcahisto_80_800_ours.sh
bash quick_start_bash/pcahisto_80_800_uniform.sh
```

* You can also override options in commandline, for example  ```python3 train_deform.py TRAIN.num_epoch 10 ```.


### Evaluation
1. Evaluate a trained model on the validation set, simply override following options ```TRAIN.start_epoch 125 TRAIN.num_epoch 126 TRAIN.eval_per_epoch 1 TRAIN.skip_train_for_eval True```

* Alternatively, you can quick start with provided bash script:
```bash
bash quick_start_bash/eval/cityscape_64_128_ours.sh
bash quick_start_bash/eval/cityscape_64_128_uniform.sh
bash quick_start_bash/eval/deepglob_300_300_ours.sh
bash quick_start_bash/eval/deepglob_300_300_uniform.sh
bash quick_start_bash/eval/pcahisto_80_800_ours.sh
bash quick_start_bash/eval/pcahisto_80_800_uniform.sh
```
