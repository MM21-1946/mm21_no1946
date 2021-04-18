import torch
from torch.functional import F 

class TanLoss(object):
    def __init__(self, min_iou, max_iou, mask2d, dataset_train=None, num_clips=None):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.num_clips = num_clips
        self.mask2d = mask2d
        """
        if dataset_train:
            self.frequency = 2*torch.ones((num_clips, num_clips), device='cuda')
            for idx in range(len(dataset_train)):
                duration = dataset_train.get_duration(idx)
                moment = dataset_train.get_moment(idx) / duration
                moment = (moment * num_clips).long()
                self.frequency[moment[0], moment[1]-1] += 1
            self.ilf_sum = torch.log(self.frequency).masked_select(self.frequency!=0).sum()
        """

    def scale(self, iou):
        # Very Important hyperparameters
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d):
        B = ious2d.shape[0]
        ious2d = self.scale(ious2d).clamp(0, 1) 
        """
        loss = F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d),
            reduction='none'
        ).reshape(B,-1)
        sample_weight = torch.empty(B, device='cuda')
        for b in range(B):
            iou2d = ious2d[b]
            indices = torch.nonzero((iou2d==torch.max(iou2d)))[0]
            # Inverted frequency
            sample_weight[b] = torch.log(self.ilf_sum / torch.log(self.frequency[indices[0], indices[1]]))
        loss = loss * sample_weight[:,None]
        loss = loss.mean()
        """
        loss = F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d)
        )
        
        return loss

        
def build_loss(cfg, arch, mask2d, dataset_train=None):
    if arch == 'TAN':
        num_clips = cfg.DATASETS.NUM_CLIPS
        min_iou = cfg.MODEL.TAN.LOSS.MIN_IOU 
        max_iou = cfg.MODEL.TAN.LOSS.MAX_IOU
        return TanLoss(min_iou, max_iou, mask2d, dataset_train, num_clips) 
    else:
        raise  NotImplementedError