import random
import torch
from torch import nn
from torch.functional import F 

class BoundaryRegressionLoss(nn.Module):
    def __init__(self):
        super(BoundaryRegressionLoss, self).__init__()
        self.regloss1 = nn.SmoothL1Loss()
        self.regloss2 = nn.SmoothL1Loss()

    def forward(self, loc, s_gt, e_gt):
        """
        Args:
            loc: location; [B,2] - start/end
            s_gt: grounding truth normalized start loc; [B] float tensor
            e_gt: grounding truth normalized end loc; [B] float tensor
        Returns:
            loss: loss value; [1], float tensor
        """
        loss = self.regloss1(loc[:,0], s_gt) + self.regloss2(loc[:,1], e_gt)
        return loss

class MomentClassificationLoss(nn.Module):
    def __init__(self, tag_weight):
        super(MomentClassificationLoss, self).__init__()
        self.w = tag_weight

    def forward(self, moment_score, targetmask):
        """ loss function to compute weighted Binary Cross-Entropy loss
        Args:
            moment_score: location; [B,num_segment] 
            targetmask: grounding truth; [B,num_segment] int tensor
        Returns:
            loss: loss value; [1], float tensor
        """
        ac_loss = (-targetmask*torch.log(moment_score+1e-8)).sum(1) / targetmask.sum(1)
        ac_loss = (self.w * ac_loss.mean(0))
        return ac_loss

class DQALoss(nn.Module):
    def __init__(self, dqa_weight, dqa_lambda):
        super(DQALoss, self).__init__()
        self.w = dqa_weight # 1.0
        self.r = dqa_lambda # 0.2

    def forward(self, attw):
        """ loss function to diversify attention weights (regularization)
        Args:
            net_outs: dictionary of network outputs
        Returns:
            loss: loss value; [1], float tensor
        """
        NA = attw.size(1)
        attw_T = torch.transpose(attw, 1, 2).contiguous()
        I = torch.eye(NA).unsqueeze(0).type_as(attw) * self.r
        P = torch.norm(torch.bmm(attw, attw_T) - I, p="fro", dim=[1,2], keepdim=True)
        da_loss = self.w * (P**2).mean()

        return da_loss

class LGILoss(nn.Module):
    def __init__(self, use_tag_loss, tag_weight, use_dqa_loss, dqa_weight=None, dqa_lambda=None):
        super(LGILoss, self).__init__()
        self.use_tag_loss = use_tag_loss
        self.use_dqa_loss = use_dqa_loss

        self.regression_loss = BoundaryRegressionLoss()
        if use_tag_loss:
            self.tag_loss = MomentClassificationLoss(tag_weight)
        if use_dqa_loss:
            self.dqa_loss = DQALoss(dqa_weight, dqa_lambda)

    def forward(self, predicted_boundarys, temporal_attention_weights, semantic_attention_weights, targets):
        """
        Args:
            predicted_boundarys (tensor[B, 2])
            semantic_attention_weights (tensor[B, nse, Lmax])
            targets (VMRTarget)
                targets.ious2d (tensor[B, nclip, nclip])
                targets.s_pos_normed (tensor[B])
                targets.e_pos_normed (tensor[B])
                targets.targetmask: (tensor[B, seg])
        Returns:
            loss_dict
        """
        loss_dict = {}
        regression_loss = self.regression_loss(predicted_boundarys, targets.s_pos_normed, targets.e_pos_normed)
        loss_dict['regression_loss'] = regression_loss
        if self.use_tag_loss:
            tag_loss = self.tag_loss(temporal_attention_weights, targets.targetmask)
            loss_dict['tag_loss'] = tag_loss
        if self.use_dqa_loss:
            dqa_loss = self.dqa_loss(semantic_attention_weights)
            loss_dict['dqa_loss'] = dqa_loss
        total_loss = sum(loss for loss in loss_dict.values())
        loss_dict['total_loss'] = total_loss
        return loss_dict


class CMINLoss(nn.Module):
    def __init__(self, alpha, high_score_thr=0.7):
        super().__init__()
        self.high_score_thr = 0.7
        self.clearing_thr = 0.3
        self.alpha = alpha
        self.alignment_loss = nn.BCELoss()
        self.regression_loss = nn.SmoothL1Loss()
    
    def forward(self, iou_predict, box_offset, ious1d, gt):
        B = iou_predict.shape[0]
        reg_indices = torch.where(ious1d>=self.high_score_thr) #(tensor, tensor)
        ious1d[ious1d<self.clearing_thr] = 0.0
        # Classification loss
        cls_loss = self.alignment_loss(iou_predict, ious1d)
        # Regression loss
        box_offset_reg = box_offset[reg_indices] # (num_cadidates, 2)
        gt = gt[reg_indices].float()
        # Total loss
        reg_loss = self.regression_loss(box_offset_reg, gt)
        loss = cls_loss + self.alpha * reg_loss
        loss_dict = {
            'total_loss': loss,
            'alignment_loss': cls_loss,
            'regression_loss': reg_loss
        }
        return loss_dict

    
class FIANLoss(nn.Module):
    def __init__(self, tau, alpha):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        
        self.alignment_loss = nn.BCELoss()
        self.regression_loss = nn.SmoothL1Loss()
    
    def forward(self, iou_predict, box_predict, ious1d, gt):
        """
        Inputs:
            iou_predict [B, num_prop]
            box_predict [B, num_prop]
            ious1d [B, num_prop]
            gt [B, 2]
        """
        B = iou_predict.shape[0]
        # Classification Loss
        cls_loss = self.alignment_loss(iou_predict, ious1d)
        # Regression Loss
        reg_indices = torch.where(ious1d>=self.tau) #(tensor, tensor)
        box_predict_reg = box_predict[reg_indices] # (num_cadidates, 2)
        gt = gt.expand(ious1d.shape[1], B, 2).transpose(0, 1).contiguous()
        gt = gt[reg_indices].float()
        reg_loss = self.regression_loss(box_predict_reg, gt)
        # Total Loss
        loss = cls_loss + self.alpha * reg_loss
        loss_dict = {
            'total_loss': loss,
            'alignment_loss': cls_loss,
            'regression_loss': reg_loss
        }
        return loss_dict


def BPRLoss(
        score_predict: torch.Tensor,
        score_real: torch.Tensor,
):
    """
    Calculate the loss of ranknet without weight
    :param score_predict: BxN tensor with model output score
    :param score_real: BxN tensor with real score
    :return: Loss of ranknet
    """
    B, N = score_real.size()
    score_diff = torch.sigmoid(score_predict[:,None,:] - score_predict[:,:,None]) # [B, N, N]
    tij = (1.0 + torch.sign(score_real[:,None,:] - score_real[:,:,None])) / 2.0 # [B, N, N]
    loss_mat = tij * torch.log(score_diff) + (1.0-tij)*torch.log(1-score_diff) # [B, N, N]
    return -loss_mat.mean()
     

def build_loss(cfg, arch):
    if arch == 'LGI':
        use_tag_loss = cfg.MODEL.LGI.LOSS.USE_TAG_LOSS
        tag_weight = cfg.MODEL.LGI.LOSS.TAG_WEIGHT
        use_dqa_loss = cfg.MODEL.LGI.LOSS.USE_DQA_LOSS
        dqa_weight = cfg.MODEL.LGI.LOSS.DQA_WEIGHT
        dqa_lambda = cfg.MODEL.LGI.LOSS.DQA_LAMBDA
        return LGILoss(
            use_tag_loss, tag_weight,
            use_dqa_loss, dqa_weight=dqa_weight, dqa_lambda=dqa_lambda
        )
    elif arch == 'CMIN':
        alpha = cfg.MODEL.CMIN.LOSS.ALPHA
        return CMINLoss(
            alpha
        )
    elif arch == 'FIAN':
        tau = cfg.MODEL.FIAN.LOSS.TAU
        alpha = cfg.MODEL.FIAN.LOSS.ALPHA
        return FIANLoss(
            tau,
            alpha,
        )
    elif arch == 'CSMGAN':
        alpha = cfg.MODEL.CSMGAN.LOSS.ALPHA
        high_score_thr = cfg.MODEL.CSMGAN.LOSS.HIGH_SCORE_THR
        return CMINLoss(
            alpha, high_score_thr
        )
    else:
        raise NotImplementedError
