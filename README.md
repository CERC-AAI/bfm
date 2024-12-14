# About

Code for ["General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data"](https://openreview.net/forum?id=HwDQH0r37I)

# Getting Started

## 0. Install the requirements

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```
or create conda environment with `conda env create -f environment.yml`
and install chronos dependency:
```
conda activate bfm
pip install "chronos-forecasting[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
```
## 1. Download and preprocess the data

Download the NMT data from [here](https://ilabel.ai/datasets/Nust-Millitary-Hospital-TUKl-NMT-EEG-Dataset/NMT-Scalp-EEG.zip) and extract it to the `data` folder. or you can use the following command:

```bash
wget https://ilabel.ai/datasets/Nust-Millitary-Hospital-TUKl-NMT-EEG-Dataset/NMT-Scalp-EEG.zip

unzip NMT-Scalp-EEG.zip -d data
```

or you can use the following command:

```bash
gdown 'https://drive.google.com/uc?id=1jD_AcmfoaIfkOiO5lSU4J6IxHZtalnTk'

unzip NMT.zip -d data/NMT/
```

## 2. Preprocess the data

To preprocess the data, run the following command:

```bash
python ./data/preprocess.py \
    --dataset nmt \
    --start_range 0 \
    --end_range 500 \
    --exp_path ./data/NMT/NMT_dl/ \
    --nmt_raw_path ./data/NMT/nmt_scalp_eeg_dataset/
```

It will preprocess the data and save it as .arrow files in the `data/NMT/nmt_dl/` folder.

## 3. Train the model

To train the model, run the following command:

```bash
accelerate launch bfm/train/train.py \
    --config bfm/configs/bfm-t5-base-nmt.yaml \
    --experiment-name "bfm-base" \
    --wandb-mode online \
    --wandb-entity <your_wandb_entity> \
    --model-id google/t5-efficient-base \
    --seed 6 \
    --learning-rate 0.001 \
    --per-device-train-batch-size 32 \
    --no-random-init \
    --n-gpus 4 \
    --max-steps 2000
```

This will train the model on the NMT dataset using the T5-base model. You can modify the config file to use a different model or dataset.

## 4. Evaluate the model

To evaluate the model, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python bfm/evaluate/evaluate.py \
    --config_path "bfm/configs/bfm-inference.yaml" \
    --directory_path "./bfm/Experiments/bfm-base_nmt" \
    --seed 2024 \
    --device "cuda"
```

[Note:] You can also use 'data/download_moabb_datasets.py' to download the MOABB datasets. Then you can use 'data/preprocess_moabb.py' to preprocess the MOABB datasets and evaluate the model on them.

# Citation

If you find this code useful, please consider citing our paper:

```
@inproceedings{
bayazi2024generalpurpose,
title={General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data},
author={Mohammad Javad Darvishi Bayazi and Hena Ghonia and Roland Riachi and Bruno Aristimunha and Arian Khorasani and Md Rifat Arefin and Amin Darabi and Guillaume Dumas and Irina Rish},
booktitle={NeurIPS Workshop on Time Series in the Age of Large Models},
year={2024},
url={https://openreview.net/forum?id=HwDQH0r37I}
}
```
