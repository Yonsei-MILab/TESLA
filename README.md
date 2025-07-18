<p align="center">
  <img src="https://img.shields.io/badge/MICCAI-2025-blueviolet.svg?style=flat-square">
  <img src="https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square">
</p>

<h1 align="center">🧠 TESLA: Test-time Reference-free Through-plane Super-resolution</h1>

<p align="center">
  <b>Official PyTorch implementation</b> of <br>
  <i>“TESLA: Test-time Reference-free Through-plane Super-resolution for Multi-contrast Brain MRI”</i> <br>
  <br>
  Accepted at <strong>MICCAI 2025</strong> 🏆
  <br><br>
  <a href="https://huggingface.co/yoonseokchoi/tesla-ckpts">Pretrained Weights</a> • 
  <a href="https://huggingface.co/datasets/yoonseokchoi/tesla-ixi">Dataset</a> • 
  <a href="#-setup-environment">Setup</a> •
  <a href="#-training-tesla">Train</a> • 
  <a href="#-testing-tesla">Test</a>  
  
</p>

---

This repository contains the official implementation of **TESLA**, a test-time reference-free through-plane super-resolution framework for multi-contrast brain MRI.


> 📄 Paper: comming soon!  
---


![TESLA ARCHITECTURE](assets/tesla_architecture.jpg)


---

## 🔧 Clone This Repository

To get started, first clone the official TESLA code from GitHub:

```bash
git clone https://github.com/yoonseokchoi-ai/TESLA
cd TESLA
```
---

## ⚙️ Setup Environment

To reproduce the TESLA pipeline, we recommend using the provided `environment.yaml` to set up a conda environment.

### 🧪 Step-by-step

```bash
# 1. Create the conda environment named 'tesla'
conda env create -f environment.yaml

# 2. Activate the environment
conda activate tesla
```
---

## 📁 Dataset: IXI (Preprocessed HDF5 format)

TESLA is trained and evaluated on a **preprocessed version of the IXI dataset**, structured as an HDF5 (`.h5`) file. The dataset has been converted into sagittal slices of shape **(N, 1, 128, 256)** where:

- `N = 4000` for training
- `N = 1000` for testing

The `.h5` file contains 9 keys:

|       Key          |          Description           |
|--------------------|--------------------------------|
|   `data_A`         |  HR T1                         | 
|   `data_B_HR`      |  HR T2                         | 
|   `data_PD`        |  HR PD                         | 
|   `data_B_21`      |  Interpolated LR T2 (x2 → x1)  |
|   `data_B_41`      |  Interpolated LR T2 (x4 → x1)  |
|   `data_B_2fold`   |  Downsampled LR T2 (x2)        |
|   `data_B_4fold`   |  Downsampled LR T2 (x4)        |
|   `data_B_SR_2to1` |  SR T2 from PR stage (x2 → x1) |
|   `data_B_SR_4to2` |  SR T2 from PR stage (x4 → x2) |



## 📦 Download Dataset (Hugging Face)

The full dataset is hosted on Hugging Face and can be downloaded with the following steps:

### 🛠 Recommended Method (entire folder)

```bash
# 1. Install git-lfs (only required once)
git lfs install

# 2. Clone the dataset into a folder named 'data'
git clone https://huggingface.co/datasets/yoonseokchoi/tesla-ixi data

# 3. Navigate into the folder
cd data

# 4. Pull large files (e.g., .h5) tracked by git-lfs
git lfs pull

```

## 🧠 Pretrained Weights on IXI (Hugging Face)

TESLA is pretrained on the above IXI dataset, and the pretrained model weights are publicly available on Hugging Face.

To download the entire ckpts/ folder (which includes all pretrained weights), follow the steps below:

```bash

# 1. Install git-lfs if not already
git lfs install

# 2. Clone the checkpoint repo into ./ckpts
git clone https://huggingface.co/yoonseokchoi/tesla-ckpts ckpts

# 3. Navigate and pull large files
cd ckpts

# 4. Pull large files (e.g., .pt) tracked by git-lfs
git lfs pull

```
```bash
./ckpts/
├── contentnet/
│   └── gen_0100.pt
├── tesla/
│   └── _tesla_reproducibility_check/
│       └── gen_100.pt
```

You can now use the checkpoints directly in your TESLA pipeline.

---

## 🚀 Training TESLA

To train TESLA on the preprocessed IXI dataset, use the following command:

```bash

python TESLA/train_h5_tesla.py -e 100 -b 5 --nb_train_imgs 4000 --nb_test_imgs 10 --gpu 0 --tb_display_size 3 --train_dataset True --crf_domain t1 --sr_scale 4 --tb_comment _tesla_train/
```
---

## 🧪 Testing TESLA

To evaluate a pretrained TESLA model on the test set from the IXI dataset, run the following command:

```bash
python TESLA/test_h5_tesla.py -e 100 -b 5 --nb_train_imgs 4000 --nb_test_imgs 1000 --gpu 3 --tb_display_size 3 --tb_comment _tesla_test/

---
