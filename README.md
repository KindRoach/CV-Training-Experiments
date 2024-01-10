# CV-Training-Experiments

## Introduction

There are some common image pre-processing steps in DL CV Model training practice,
which make inference application code complex and consume more compute resources.
This project targets to figure out how them impact the training results.

- Normalization or not 
- Resize keeping ratios or not 
- BGR to RGB

## Download Dataset

Download dataset from [BIRDS 525 SPECIES](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) 
and unzip them to directory `data`.

## Install Python Env

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```