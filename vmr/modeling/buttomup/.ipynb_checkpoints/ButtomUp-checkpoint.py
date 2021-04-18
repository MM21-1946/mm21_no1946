import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .VideoEncoder import build_video_encoder
from .QueryEncoder import build_query_encoder
from .MMEncoder import build_multimodal_encoder
from .Predictor import build_predictor
from .Loss import build_loss
from .BUModules import DynamicGRU
from ..Transformer import PositionalEncoding

class GeneralizedButtomUp(nn.Module):
    def __init__(self, cfg, train_dataset=None):
        super(GeneralizedButtomUp, self).__init__()
        self.arch = cfg.MODEL.ARCHITECTURE
        # build video & query encoder
        self.video_encoder = build_video_encoder(cfg, self.arch)
        self.query_encoder = build_query_encoder(cfg, self.arch)
        # build local-global video-text interactions network (vti_fn)
        self.multimodal_encoder = build_multimodal_encoder(cfg, self.arch)
        # build grounding fn
        self.prediction_head = build_predictor(cfg, self.arch)
        # build criterion
        self.loss = build_loss(cfg, self.arch)
    
    def forward(self, batches, targets=None):
        """
        Arguments:
            batches (VMRBatch)
                batches.feats (tensor[B, seg, feat_dim])
                batches.segmasks (tensor[B, seg])
                batches.queries (tensor[B, maxL, w2v_dim])
                batches.adjmats (tensor[B, maxl, maxL])
                batches.wordmasks (tensor[B, maxL])
            targets (VMRTarget)
                targets.ious2d (None)
                targets.s_pos_normed (tensor[B])
                targets.e_pos_normed (tensor[B])
                targets.targetmask (tensor[B, seg])
                targets.ious1d (tensor[B, seg*num_anchors])
                targets.ious1dmask (tensor[B, seg*num_anchors])
        """
        pass


class LGI(GeneralizedButtomUp):
    def __init__(self, cfg, dataset_train=None):
        super(LGI, self).__init__(cfg)
        self.num_semantic_entity = cfg.MODEL.LGI.QUERYENCODER.NUM_SEMANTIC_ENTITY
    
    def forward(self, batches, targets=None):
        """
        Returns:
            loc (tensor(B, 2))
        """
        wordlens = torch.sum(batches.wordmasks, dim=1).cpu()
        seg_feats = self.video_encoder(batches.feats, batches.segmasks)
        # get semantic phrase features:
        # se_feats: semantic phrase features [B,nse,*];
        #           ([e^1,...,e^n]) in Eq. (7)
        # se_attw: attention weights for semantic phrase [B,nse,Lmax];
        #           ([a^1,...,a^n]) in Eq. (6)
        query_encoded_dict = self.query_encoder(batches.queries, wordlens, batches.wordmasks)
        # Local-global video-text interactions
        # sa_feats: semantics-aware segment features [B,nseg,d]; R in Eq. (12)
        # s_attw: aggregating weights [B,nse]
        if self.num_semantic_entity > 1:
            q_feats, se_attw = query_encoded_dict['semantic_features'], query_encoded_dict['semantic_attention_weights']
        else:
            q_feats, se_attw = query_encoded_dict['sentence_features'], None
        semantic_aware_seg_feats, s_attw = self.multimodal_encoder(seg_feats, batches.segmasks, q_feats)
        # Temporal attentive localization by regression
        # loc: prediction of time span (t^s, t^e)
        # t_attw: temporal attention weights (o)
        loc, t_attw = self.prediction_head(semantic_aware_seg_feats, batches.segmasks)
        #   loc: prediction of normalized time span (t^s, t^e); [B,2]
        #   att_w: temporal attention weights (o); [B,seg]

        if self.training:
            loss_dict = self.loss(loc, t_attw, se_attw, targets)
            return loss_dict
        else:
            return loc.clamp(0., 1.).unsqueeze(1), torch.tensor(1.0).expand(loc.shape[0], 1) # normalized time span


class CMIN(GeneralizedButtomUp):
    def __init__(self, cfg, dataset_train=None):
        super().__init__(cfg)
        self.proposals, self.num_anchors, self.anchor_widths, self.iou1dmask = self._generate_proposal(cfg)
        if cfg.MODEL.CAUSAL.USE_CAUSALITY:
            self.normalized_predictor = True
        else:
            self.normalized_predictor = False
    
    @staticmethod
    def _generate_proposal(cfg):
        widths = np.array(cfg.DATASETS.ANCHOR_WIDTHS)
        num_segments = cfg.INPUT.NUM_SEGMENTS
        anchor_widths = torch.tensor(widths, device='cuda')[None,:]
        anchor_widths = anchor_widths.expand(num_segments,len(widths)).reshape(-1).float() # [video_len*num_anchors]
    
        center = 7.5
        start = center - 0.5 * (widths - 1)
        end = center + 0.5 * (widths - 1)
        anchors = np.stack([start, end], -1)
        num_anchors = anchors.shape[0]
        widths = (anchors[:, 1] - anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, num_segments)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]
        
        proposals_flatten = np.reshape(proposals, [-1, 2]) # [video_len*num_anchors, 2]
        illegal = np.logical_or(proposals_flatten[:, 0] < 0, proposals_flatten[:, 1] >= num_segments)
        scores_mask = (1 - illegal).astype(np.uint8)
        scores_mask = torch.from_numpy(scores_mask).unsqueeze(0).to('cuda') # [1, video_len*num_anchors]
        
        proposals = torch.from_numpy(proposals).float().to('cuda').reshape(-1, 2)  # [video_len*num_anchors, 2]

        return proposals, num_anchors, anchor_widths, scores_mask
    
    def forward(self, batches, targets=None):
        """
        Returns:
            loc_norm_predict (tensor(B, seg*num_anchors, 2))
            iou_predict (tensor(B, seg*num_anchors))
        """
        num_seg = batches.feats.shape[1]
        seglens = batches.segmasks.sum(dim=-1)
        querylens = batches.wordmasks.sum(dim=-1)
        # encoder
        self_attentive_segfeats = self.video_encoder(batches.feats, seglens) # [B, maxL, hdim]
        syntactic_aware_wordfeats = self.query_encoder(batches.queries, batches.wordmasks, querylens, batches.adjmats) # [B, seg, hdim]
        # interactive
        crossmodal_feats = self.multimodal_encoder(self_attentive_segfeats, seglens, syntactic_aware_wordfeats, batches.wordmasks)
        # loss
        iou_predict, box_offset = self.prediction_head(crossmodal_feats, self.iou1dmask) # [B, seg*num_anchors], [B, seg*num_anchors, 2]
        if self.training:
            if not self.normalized_predictor:
                # [B, 2] - [seg*num_anchors, 2]
                gt = torch.stack([targets.s_pos_normed*num_seg, targets.e_pos_normed*num_seg], dim=1)
                gt = gt[:, None, :] - self.proposals[None,:,:]
                loss_dict = self.loss(iou_predict, box_offset, targets.ious1d, gt)
                return loss_dict
            else:
                # [B, 2] - [seg*num_anchors, 2]
                gt = torch.stack([targets.s_pos_normed*num_seg, targets.e_pos_normed*num_seg], dim=1)
                gt_center = gt[:,None,:].mean(dim=2) - self.proposals[None,:,:].mean(dim=2) # [B, seg*num_anchors]
                gt_center = gt_center / self.anchor_widths[None,:] # [B, seg*num_anchors]
                gt_width = torch.log((gt[:,None,1] - gt[:,None,0]) / self.anchor_widths[None,:]) # [B, seg*num_anchors]
                gt_offset = torch.stack([gt_center, gt_width], dim=2)
                loss_dict = self.loss(iou_predict, box_offset, targets.ious1d, gt_offset)
                return loss_dict
        else:
            if not self.normalized_predictor:
                box_anchor = self.proposals[None,:,:]
                box_predict = box_anchor + box_offset # [B, seg*num_anchors, 2]
                loc_norm_predict = (box_predict/num_seg).clamp(0., 1.)
                return loc_norm_predict, iou_predict
            else:
                center_offset = box_offset[:,:,0]*self.anchor_widths[None,:] # [B, seg*num_anchors]
                width = torch.exp(box_offset[:,:,1])*self.anchor_widths[None,:] # [B, seg*num_anchors]
                center = self.proposals[None,:].mean(dim=2) + center_offset # [B, seg*num_anchors]
                box_predict = torch.stack([center-0.5*width, center+0.5*width], dim=2) # [B, seg*num_anchors, 2]
                loc_norm_predict = (box_predict/num_seg).clamp(0., 1.)
                return loc_norm_predict, iou_predict


class FIAN(GeneralizedButtomUp):
    def __init__(self, cfg, dataset_train=None):
        super().__init__(cfg)
        self.proposals = self._generate_proposal(cfg)
    
    @staticmethod
    def _generate_proposal(cfg):
        widths = np.array(cfg.DATASETS.WINDOW_WIDTHS)
        overlap = np.array(cfg.DATASETS.WINDOW_OVERLAP)
        num_segments = cfg.INPUT.NUM_SEGMENTS

        strides = (widths*(1-overlap)).astype(int)
        prop_lens = ((num_segments-widths)/strides+1).astype(int)
        proposal = []
        for prop_len, width, stride in zip(prop_lens, widths, strides):
            for k in range(prop_len):
                proposal.append([k*stride, k*stride+width-1])
        return torch.tensor(proposal, device='cuda')
    
    def forward(self, batches, targets=None):
        """
        Returns:
            loc_norm_predict (tensor(B, seg*num_anchors, 2))
            iou_predict (tensor(B, seg*num_anchors))
        """
        B, num_seg, _ = batches.feats.size()
        maxL = batches.queries.size(1)
        seglens = batches.segmasks.sum(dim=-1)
        querylens = batches.wordmasks.sum(dim=-1)
        # encoder
        segfeats = self.video_encoder(batches.feats, seglens)
        queryfeats = self.query_encoder(batches.queries, querylens)
        # interactive
        crossmodal_dicts = self.multimodal_encoder(segfeats, queryfeats, seglens, batches.wordmasks)
        frame_embed = crossmodal_dicts['frame_embed'] # [B, num_seg, hdim]
        # loss
        iou_predict, box_predict = self.prediction_head(frame_embed, self.proposals) # [B, num_anchors], [B, num_anchors, 2]
        if self.training:
            gt = torch.stack([targets.s_pos_normed*num_seg, targets.e_pos_normed*num_seg], dim=1)
            loss_dict = self.loss(iou_predict, box_predict, targets.ious1d, gt)
            return loss_dict
        else:
            loc_norm_predict = (box_predict/num_seg).clamp(0., 1.)
            return loc_norm_predict, iou_predict

class CSMGAN(GeneralizedButtomUp):
    def __init__(self, cfg, dataset_train=None):
        super().__init__(cfg)
        self.proposals, self.num_anchors, self.anchor_widths, self.iou1dmask = self._generate_proposal(cfg)
        if cfg.MODEL.CAUSAL.USE_CAUSALITY:
            self.normalized_predictor = True
        else:
            self.normalized_predictor = False
    
    @staticmethod
    def _generate_proposal(cfg):
        widths = np.array(cfg.DATASETS.ANCHOR_WIDTHS)
        num_segments = cfg.INPUT.NUM_SEGMENTS
        anchor_widths = torch.tensor(widths, device='cuda')[None,:]
        anchor_widths = anchor_widths.expand(num_segments,len(widths)).reshape(-1).float() # [video_len*num_anchors]
    
        center = 7.5
        start = center - 0.5 * (widths - 1)
        end = center + 0.5 * (widths - 1)
        anchors = np.stack([start, end], -1)
        num_anchors = anchors.shape[0]
        widths = (anchors[:, 1] - anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, num_segments)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]
        
        proposals_flatten = np.reshape(proposals, [-1, 2]) # [video_len*num_anchors, 2]
        illegal = np.logical_or(proposals_flatten[:, 0] < 0, proposals_flatten[:, 1] >= num_segments)
        scores_mask = (1 - illegal).astype(np.uint8)
        scores_mask = torch.from_numpy(scores_mask).unsqueeze(0).to('cuda') # [1, video_len*num_anchors]
        
        proposals = torch.from_numpy(proposals).float().to('cuda').reshape(-1, 2)  # [video_len*num_anchors, 2]

        return proposals, num_anchors, anchor_widths, scores_mask
    
    def forward(self, batches, targets=None):
        """
        Returns:
            loc_norm_predict (tensor(B, seg*num_anchors, 2))
            iou_predict (tensor(B, seg*num_anchors))
        """
        num_seg = batches.feats.shape[1]
        seglens = batches.segmasks.sum(dim=-1)
        querylens = batches.wordmasks.sum(dim=-1)
        # encoder
        self_attentive_segfeats = self.video_encoder(batches.feats, seglens)
        syntactic_aware_wordfeats = self.query_encoder(batches.queries, querylens)
        # interactive
        crossmodal_feats = self.multimodal_encoder(self_attentive_segfeats, seglens, syntactic_aware_wordfeats, batches.wordmasks)
        # loss
        iou_predict, box_offset = self.prediction_head(crossmodal_feats, self.iou1dmask) # [B, seg*num_anchors], [B, seg*num_anchors, 2]
        if self.training:
            if not self.normalized_predictor:
                # [B, 2] - [seg*num_anchors, 2]
                gt = torch.stack([targets.s_pos_normed*num_seg, targets.e_pos_normed*num_seg], dim=1)
                gt = gt[:, None, :] - self.proposals[None,:,:]
                loss_dict = self.loss(iou_predict, box_offset, targets.ious1d, gt)
                return loss_dict
            else:
                # [B, 2] - [seg*num_anchors, 2]
                gt = torch.stack([targets.s_pos_normed*num_seg, targets.e_pos_normed*num_seg], dim=1)
                gt_center = gt[:,None,:].mean(dim=2) - self.proposals[None,:,:].mean(dim=2) # [B, seg*num_anchors]
                gt_center = gt_center / self.anchor_widths[None,:] # [B, seg*num_anchors]
                gt_width = torch.log((gt[:,None,1] - gt[:,None,0]) / self.anchor_widths[None,:]) # [B, seg*num_anchors]
                gt_offset = torch.stack([gt_center, gt_width], dim=2)
                loss_dict = self.loss(iou_predict, box_offset, targets.ious1d, gt_offset)
                return loss_dict
        else:
            if not self.normalized_predictor:
                box_anchor = self.proposals[None,:,:]
                box_predict = box_anchor + box_offset # [B, seg*num_anchors, 2]
                loc_norm_predict = (box_predict/num_seg).clamp(0., 1.)
                return loc_norm_predict, iou_predict
            else:
                center_offset = box_offset[:,:,0]*self.anchor_widths[None,:] # [B, seg*num_anchors]
                width = torch.exp(box_offset[:,:,1])*self.anchor_widths[None,:] # [B, seg*num_anchors]
                center = self.proposals[None,:].mean(dim=2) + center_offset # [B, seg*num_anchors]
                box_predict = torch.stack([center-0.5*width, center+0.5*width], dim=2) # [B, seg*num_anchors, 2]
                loc_norm_predict = (box_predict/num_seg).clamp(0., 1.)
                return loc_norm_predict, iou_predict