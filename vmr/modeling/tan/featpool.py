import torch
from torch import nn
import torch.nn.functional as F

class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride, freeze):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.pool(self.conv(x.transpose(1, 2)).relu())
        

def build_featpool(cfg, arch):
    input_dim = cfg.DATASETS.VISUAL_DIM
    if arch == 'TAN':
        input_size = cfg.MODEL.TAN.FEATPOOL.INPUT_SIZE
        hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
        kernel_size = cfg.MODEL.TAN.FEATPOOL.KERNEL_SIZE
        stride = cfg.INPUT.NUM_SEGMENTS // cfg.DATASETS.NUM_CLIPS
        freeze = cfg.MODEL.TAN.FEATPOOL.FREEZE
        return FeatAvgPool(input_size, hidden_size, kernel_size, stride, freeze)
    else:
        raise NotImplementedError