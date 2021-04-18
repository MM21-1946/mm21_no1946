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


class ActivityNetDataset(VMRDataset):
    def __init__(self, **kwargs):
        super(ActivityNetDataset, self).__init__(dataset_name='activitynet', **kwargs)
