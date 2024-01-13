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
and unzip it to directory `data/birds525`.

Download dataset from [Caltech 256 Image Dataset](https://www.kaggle.com/datasets/jessicali9530/caltech256/data)
and unzip it to directory `data/objects256/images`.

## Install Python Env

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```