import torch
from torch import nn
from torch.functional import F

from .BUModules import AttentivePooling


class LGIPredictor(nn.Module):
    def __init__(self, visual_hidden_dim, grounding_hidden_dim, num_segment, logic_head, causal, tau, gamma, alpha):
        super(LGIPredictor, self).__init__()
        self.causal = causal
        self.tau = tau
        self.gamma = gamma# * torch.ones((1,self.num_anchors,1), device='cuda', requires_grad=False)
        self.alpha = alpha
        
        if self.causal:
            self.register_buffer('d', torch.zeros((1,grounding_hidden_dim), device='cuda', requires_grad=False))
            self.cos_sim = nn.CosineSimilarity(dim=1)
        
        self.tatt = AttentivePooling(
            num_layer=1, 
            feat_dim=visual_hidden_dim, 
            hidden_dim=visual_hidden_dim//2, 
            use_embedding=True, 
            embedding_dim=visual_hidden_dim
        )
        # Regression layer
        nn_list = [
            nn.Linear(grounding_hidden_dim, grounding_hidden_dim), 
            nn.ReLU(), 
            nn.Linear(grounding_hidden_dim, 2)
        ]
        nn_list.append(getattr(nn, logic_head)()) # ReLU for charades, Sigmoid for anet
        self.MLP_reg = nn.Sequential(*nn_list)

    def forward(self, semantic_aware_seg_feats, seg_masks):
        """ Perform Regression
        Args:
            semantic_aware_seg_feats: segment-level features; [B,seg,D]
            seg_masks: masks for effective segments in video; [B,seg]
        Returns:
            loc: prediction of normalized time span (t^s, t^e); [B,2]
            att_w: temporal attention weights (o); [B,seg]
        """
        if self.causal:
            # De-confound Training
            semantic_aware_seg_feats = F.normalize(semantic_aware_seg_feats, dim=2)
            summarized_vfeat, att_w = self.tatt(semantic_aware_seg_feats, seg_masks)
            if self.training:
                self.d = 0.9*self.d + 0.1*summarized_vfeat.mean(0, keepdim=True) # [1, hdim]
            else:
                # counterfactual TDE inference
                bias = self.cos_sim(summarized_vfeat, self.d).unsqueeze(1) * self.d # [1, hdim]
                summarized_vfeat = summarized_vfeat - self.alpha*bias
        else:
            # perform Eq. (13) and (14)
            summarized_vfeat, att_w = self.tatt(semantic_aware_seg_feats, seg_masks)
        # perform Eq. (15)
        loc = self.MLP_reg(summarized_vfeat) # loc = [t^s, t^e]
        return loc, att_w

class CMINPredictor(nn.Module):
    def __init__(self, num_segment, model_hidden_dim, anchors, causal, tau, gamma, alpha):
        super().__init__()
        self.num_anchors = len(anchors)
        self.causal = causal
        self.tau = tau
        self.gamma = gamma # * torch.ones((1,self.num_anchors,1), device='cuda', requires_grad=False)
        self.alpha = alpha
        if self.causal:
            self.register_buffer('d', torch.zeros((1,num_segment,model_hidden_dim), device='cuda', requires_grad=False))
            self.cos_sim = nn.CosineSimilarity(dim=2)
            self.fc_score = nn.Conv1d(model_hidden_dim, self.num_anchors, kernel_size=1, padding=0, stride=1, bias=False)
        else:
            self.fc_score = nn.Conv1d(model_hidden_dim, self.num_anchors, kernel_size=1, padding=0, stride=1)
        self.fc_reg = nn.Conv1d(model_hidden_dim, self.num_anchors << 1, kernel_size=1, padding=0, stride=1)

    def forward(self, mmfeats, ious1dmask):
        """
        Inputs:
            mmfeats (tensor[B, seg, hdim])
        Returns:
            iou_predict (tensor(B, seg*num_anchors))
            box_predict (tensor(B, seg*num_anchors, 2))
            ious1dmask (tensor(1, seg*num_anchors))
        """
        B, seg = mmfeats.shape[0], mmfeats.shape[1]
        # Predict Alignment Score
        if self.causal:
            if self.training:
                # De-confound Training
                self.d = 0.9*self.d + 0.1*mmfeats.detach().mean(0, keepdim=True) # [1, seg, hdim]
                mmfeats = F.normalize(mmfeats, dim=2)
                iou_predict = self.tau * self.fc_score(mmfeats.transpose(-1, -2)) / \
                                (torch.norm(self.fc_score.weight[:,:,0],dim=1)[None,:,None] + self.gamma) # [B, num_anchors, seg]
            else:
                # counterfactual TDE inference
                bias = self.cos_sim(mmfeats, self.d).unsqueeze(2) * F.normalize(self.d, dim=2) # [1, seg, hdim]
                mmfeats = F.normalize(mmfeats, dim=2) - self.alpha*bias
                iou_predict = self.tau * self.fc_score(mmfeats.transpose(-1, -2)) / \
                                (torch.norm(self.fc_score.weight[:,:,0],dim=1)[None,:,None] + self.gamma) # [B, num_anchors, seg]
        else:
            iou_predict = self.fc_score(mmfeats.transpose(-1, -2)) # [B, seg, num_anchors]
        iou_predict = torch.sigmoid(iou_predict).transpose(-1, -2) # [B, seg, num_anchors]
        iou_predict = iou_predict.contiguous().view(B, -1) * ious1dmask.float()
        # Predict Classification Score
        box_offset = self.fc_reg(mmfeats.transpose(-1, -2)).transpose(-1, -2)
        box_offset = box_offset.contiguous().view(B, seg * self.num_anchors, 2) # [B, seg*num_anchor, 2]
        
        return iou_predict, box_offset


class FIANPredictor(nn.Module):
    def __init__(self, num_segment, model_hidden_dim, anchors, overlap, causal, tau, gamma, alpha):
        super().__init__()
        self.num_anchors = len(anchors)
        self.causal = causal
        self.tau = tau
        self.gamma = gamma # * torch.ones((1,self.num_anchors,1), device='cuda', requires_grad=False)
        self.alpha = alpha
        
        self.norm = nn.Sequential(
            nn.BatchNorm1d(model_hidden_dim),
            nn.Dropout(0.2)
        )
        self.fc_score = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        
        if self.causal:
            self.register_buffer('d', torch.zeros((1, model_hidden_dim, num_segment), device='cuda', requires_grad=False))
            self.cos_sim = nn.CosineSimilarity(dim=1)
        
        for width in anchors:
            stride = int((1-overlap)*width)
            self.fc_score.append(
                nn.Conv1d(model_hidden_dim, 1, kernel_size=width, stride=stride, bias=~self.causal)
            )
            self.fc_reg.append(
                nn.Conv1d(model_hidden_dim, 2, kernel_size=width, stride=stride)
            )

    def forward(self, mmfeats, proposals):
        """
        Inputs:
            mmfeats (tensor(B, seg, hdim))
            proposals  (tensor(num_prop, 2))
        Returns:
            iou_predict (tensor(B, seg*num_anchors))
            box_predict (tensor(B, seg*num_anchors, 2))
            ious1dmask (tensor(1, seg*num_anchors))
        """
        B, seg = mmfeats.shape[0], mmfeats.shape[1]
        mmfeats = self.norm(mmfeats.transpose(-1, -2)) # [B, hdim, seg]
        iou_predict = []
        box_offset = []
        
        for k in range(self.num_anchors):
            # Predict Alignment 
            if self.causal:
                if self.training:
                    self.d = 0.9*self.d + 0.1*mmfeats.detach().mean(0, keepdim=True) # [1, hdim, seg]
                    mmfeats = F.normalize(mmfeats, dim=1)
                    iou_predict.append(
                        self.tau * self.fc_score[k](mmfeats).squeeze(1) / (torch.norm(self.fc_score[k].weight) + self.gamma)
                    )
                else:
                    bias = self.cos_sim(mmfeats, self.d).unsqueeze(1) * F.normalize(self.d, dim=1)
                    mmfeats = F.normalize(mmfeats, dim=1)
                    iou_predict.append(
                        self.tau * self.fc_score[k](mmfeats - self.alpha*bias).squeeze(1) / (torch.norm(self.fc_score[k].weight) + self.gamma)
                    )
            else:
                iou_predict.append(
                    self.fc_score[k](mmfeats).squeeze(1)
                ) # [B, num_prop_width]
            # Predict Classification Score
            box_offset.append(
                self.fc_reg[k](mmfeats)
            ) # [B, 2, num_prop_width]
        # Predict Classification Score
        iou_predict = torch.cat(iou_predict, dim=1)
        iou_predict = torch.sigmoid(iou_predict) # [B, num_prop]
        # Predict Alignment Score
        box_offset = torch.cat(box_offset, dim=2)
        box_offset = box_offset.transpose(-1, -2)  # [B, num_prop, 2]
        box_anchor = proposals
        box_predict = box_anchor + box_offset # [B, num_prop, 2]

        return iou_predict, box_predict


def build_predictor(cfg, arch):
    causal = cfg.MODEL.CAUSAL.USE_CAUSALITY
    tau = cfg.MODEL.CAUSAL.TAU
    gamma = cfg.MODEL.CAUSAL.GAMMA
    alpha = cfg.MODEL.CAUSAL.ALPHA
    video_segment_num = cfg.INPUT.NUM_SEGMENTS
    if arch == 'LGI':
        visual_hidden_dim = cfg.MODEL.LGI.VIDEOENCODER.HIDDEN_SIZE
        grounding_hidden_dim = cfg.MODEL.LGI.PREDICTOR.GROUNDING_HDIM
        logic_head = cfg.MODEL.LGI.PREDICTOR.LOGIC_HEAD
        return LGIPredictor(
            visual_hidden_dim,
            grounding_hidden_dim,
            video_segment_num,
            logic_head, causal, tau, gamma, alpha
        )
    elif arch == 'CMIN':
        model_hidden_dim = cfg.MODEL.CMIN.HIDDEN_DIM
        anchors = cfg.DATASETS.ANCHOR_WIDTHS
        num_segment = cfg.INPUT.NUM_SEGMENTS
        return CMINPredictor(
            num_segment, model_hidden_dim, anchors, causal, tau, gamma, alpha
        )
    elif arch == 'FIAN':
        model_hidden_dim = cfg.MODEL.FIAN.MMENCODER.HIDDEN_DIM
        anchors = cfg.DATASETS.WINDOW_WIDTHS
        overlap = cfg.DATASETS.WINDOW_OVERLAP
        return FIANPredictor(
            video_segment_num,
            model_hidden_dim,
            anchors,
            overlap, causal, tau, gamma, alpha
        )
    elif arch == 'CSMGAN':
        model_hidden_dim = cfg.MODEL.CSMGAN.HIDDEN_DIM
        anchors = cfg.DATASETS.ANCHOR_WIDTHS
        num_segment = cfg.INPUT.NUM_SEGMENTS
        return CMINPredictor(
            num_segment, model_hidden_dim, anchors, causal, tau, gamma, alpha
        )
    else:
        raise NotImplementedError