import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vmr.structures import VMRBatch, VMRGroundTruth


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        idxs, feats, seglens, queries, querymasks, adjmats, constimasks, wordlens, gt_dict_list = transposed_batch
        B, num_segment = len(feats), feats[0].shape[0]
        # Build word mask and segment mask
        max_word_length = max([query.shape[0] for query in queries])
        wordmasks = torch.zeros(B, max_word_length, dtype=torch.int32)
        for b, wordlen in enumerate(wordlens):
             wordmasks[b, :wordlen] = querymasks[b]
        segmasks = torch.zeros(B, num_segment)
        for b, seglen in enumerate(seglens):
            segmasks[b, :seglen] = 1.0
        # Build dependency parse tree adjacent matrix if exists
        if adjmats[0] is not None:
            adjmat_list = []
            for adjmat in adjmats:
                padding = wordmasks.shape[1] - adjmat.shape[0]
                adjmat_list.append(
                    F.pad(adjmat, (0,padding,0,padding), "constant", 0)
                )
            adjmats = torch.stack(adjmat_list)
        else:
            adjmats = None
        
        if constimasks[0] is not None:
            constimask_list = []
            for constimask in constimasks:
                padding = wordmasks.shape[1] - constimask.shape[1]
                constimask_list.append(
                    F.pad(constimask, (0,padding,0,padding), "constant", 0)
                )
            constimasks = torch.stack(constimask_list)
        else:
            constimasks = None
        
        # Build gt dict
        gt_dict = {}
        for key in gt_dict_list[0]:
            gt_dict[key] = torch.stack([d[key] for d in gt_dict_list])
        if 'start_pos_norm' in gt_dict:
            # Build annotaion mask
            s_pos_normed = [d['start_pos_norm'] for d in gt_dict_list]
            e_pos_normed = [d['end_pos_norm'] for d in gt_dict_list]
            targetmask = torch.zeros(B, num_segment)
            for b in range(B):
                targetmask[b, int(s_pos_normed[b]*seglens[b]):int(e_pos_normed[b]*seglens[b])+1] = 1.0
            gt_dict['targetmask'] = targetmask
        
        # Reture batch and annotations
        return_turple = (
            VMRBatch(
                feats=torch.stack(feats).float(),
                segmasks=segmasks,
                queries=pad_sequence(queries).transpose(0, 1),
                adjmats=adjmats,
                constimasks=constimasks,
                wordmasks=wordmasks
            ), 
            VMRGroundTruth(
                ious2d=gt_dict['iou2d'] if 'iou2d' in gt_dict else None,
                s_pos_normed=gt_dict['start_pos_norm'] if 'start_pos_norm' in gt_dict else None,
                e_pos_normed=gt_dict['end_pos_norm'] if 'end_pos_norm' in gt_dict else None,
                targetmask=gt_dict['targetmask'] if 'targetmask' in gt_dict else None,
                ious1d=gt_dict['iou1d'] if 'iou1d' in gt_dict else None,
                ious1dmask=gt_dict['iou1d_mask'] if 'iou1d_mask' in gt_dict else None,
            ), 
            idxs
        )
        return return_turple