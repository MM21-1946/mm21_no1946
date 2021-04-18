# On Temporal Language Localizaiton Meets Long-tailed Distribution

## Introduction

This is a code base of Temporal Language Localizaiton. The framework of the code is borrowed from [2DTAN](https://github.com/ChenJoya/2dtan).

## Installation
For installation of this repository, please refer to INSTALL.md.

## Dataset
Please refer to DATASET.md in [2DTAN](https://github.com/ChenJoya/2dtan) to prepare datasets. Then, in order to generate dependency parsing tree for CMIN method, run the scripts ```scripts/generate_depedency_parsing_tree.py```.

## Reproduce Baseline
```bash
bash scripts/baseline.sh
```

## Reproduce Observations

Follow the jupyter notebook in ```scripts/bias_analysis.ipynb``` after you have reproduced baselines.

## Reproduce Normalized predictor and Normalized predictor + TDE
```bash
bash scripts/normalized.sh
bash scripts/normalized_tde.sh
```