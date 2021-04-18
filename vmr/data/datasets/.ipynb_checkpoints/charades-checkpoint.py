import os
from os.path import join, dirname
from tqdm import tqdm
import json
import pickle
import logging

import torch
from transformers import BertTokenizer, BertForPreTraining

from .vmrdataset import VMRDataset
from .utils import video2feats, moment_to_iou2d

class CharadesDataset(VMRDataset):
    def __init__(self, **kwargs):
        super(CharadesDataset, self).__init__(dataset_name='charades', **kwargs)