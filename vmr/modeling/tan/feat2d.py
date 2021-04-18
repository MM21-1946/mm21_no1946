import torch
from torch import nn
import torch.nn.functional as F

class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                # fill a diagonal line 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d


class SparseMeanPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMeanPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                # fill a diagonal line 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for (i, j) in self.maskij:
            map2d[:, :, i, j] = F.adaptive_avg_pool1d(x, len(i))
        return map2d


def build_feat2d(cfg, arch):
    num_clips = cfg.DATASETS.NUM_CLIPS
    if arch == 'TAN':
        pooling_counts = cfg.MODEL.TAN.FEAT2D.POOLING_COUNTS
        return SparseMaxPool(pooling_counts, num_clips)
    else:
        raise NotImplementedError
