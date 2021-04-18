## Installation

### Requirements:
- CUDA >= 9.0

### Installation
```bash
# create and activate a clean conda env
conda create -n vmr
conda activate vmr 

# install the right pip and dependencies for the fresh python
conda install ipython pip

# install some dependencies
pip install yacs h5py terminaltables tqdm 
pip install transformers benepar
conda install spacy

python -m spacy download en_core_web_md
# Parsing models need to be downloaded separately, using the commands:
>>> import benepar
>>> benepar.download('benepar_en3')

# follow PyTorch installation in https://pytorch.org/gect-started/locally/
# we give the instructions for CUDA 10.1, others are also okay
conda install pytorch torchvision cudatoolkit=10.1 torchtext -c pytorch