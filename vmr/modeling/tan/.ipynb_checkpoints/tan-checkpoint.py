import torch
from torch import nn
import torch.nn.functional as F

from .featpool import build_featpool
from .feat2d import build_feat2d
from .ClipProposalNetwork import build_clip_proposal_network
from .integrator import build_integrator
from .MultiModalEncoder import build_multimodal_encoder
from .predictor import build_predictor, TANPredictor
from .loss import build_loss


class TAN(nn.Module):
    def __init__(self, cfg, dataset_train=None):
        super(TAN, self).__init__()
        self.featpool = build_featpool(cfg, 'TAN') 
        self.feat2d = build_feat2d(cfg, 'TAN')
        self.integrator = build_integrator(cfg, 'TAN')
        self.predictor = build_predictor(cfg, 'TAN', self.feat2d.mask2d)
        self.tanloss = build_loss(cfg, 'TAN', self.feat2d.mask2d, dataset_train)
    
    @staticmethod
    def _score2d_to_moments_norm_and_scores(score2d, num_clips):
        """
            moments_norm # [B, num_cand, 2]
            scores # [B, num_cand]
        """
        if len(score2d.shape) == 2:
            score2d = score2d.unsqueeze(0)
        moments_norm = []
        scores = []
        for score2d_batch in score2d:
            grids = torch.nonzero(score2d_batch)
            scores.append(score2d_batch[grids[:,0], grids[:,1]])
            grids[:, 1] += 1
            moments_norm.append(grids.float() / num_clips)
        return torch.stack(moments_norm, 0) , torch.stack(scores, 0)
    
    def forward(self, batches, targets=None):
        """
        Arguments:
            batched (VMRBatch)
                batches.feats (tensor[B, seg, feat_dim])
                batches.segmasks (tensor[B, seg])
                batches.queries (tensor[B, maxL, w2v_dim])
                batches.wordmasks (tensor[B, maxL])
            targets (VMRTarget)
                targets.ious2d (tensor[B, nclip, nclip])
                targets.s_pos_normed (None)
                targets.e_pos_normed (None)
                targets.targetmask: (None)
        Returns:

        """
        wordlens = torch.sum(batches.wordmasks, dim=1)
        feats = self.featpool(batches.feats) # [B, hdim, nclip]
        map2d = self.feat2d(feats) # [B, hidm, nclip, nclip]
        map2d = self.integrator(batches.queries, wordlens, map2d) # [B, hidm, nclip, nclip]
        scores2d = self.predictor(map2d) # [B, nclip, nclip]
        # print(self.training) 
        if self.training:
            return {'total_loss': self.tanloss(scores2d, targets.ious2d)}
        else:
            scores2d = scores2d.sigmoid_() * self.feat2d.mask2d
            moments_norm , scores = self._score2d_to_moments_norm_and_scores(scores2d, scores2d.shape[1])
            return moments_norm, scores