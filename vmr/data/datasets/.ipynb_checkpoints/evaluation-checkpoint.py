from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import pickle
import os

import torch

from vmr.data import datasets
from vmr.data.datasets.utils import iou, score2d_to_moments_scores

def nms(moments, scores, topk, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed]

def iou_padding(ious_o, r):
    if ious_o.size(0) < r:
        padding = r - ious_o.size(0)
        ious = torch.nn.functional.pad(ious_o, (0, padding), "constant", 0) # [r]
    else:
        ious = ious_o # [r]
    return ious

def evaluate(dataset, predictions, nms_thresh, result_output_dir, recall_metrics=(1,5), iou_metrics=(0.1,0.3,0.5,0.7), num_chunks=5):
    """evaluate dataset using different methods based on dataset type.
        Args: 
            predictions: list[
                    (
                        moments_norm (tensor(num_predictions, 2)), 
                        scores (tensor(num_predictions))
                    )
                ]
        Returns:
            Recall@1 mIoU float
    """
    # Process dataset
    dataset_name = dataset.__class__.__name__
    frequency = torch.zeros((num_chunks, num_chunks))
    for idx in range(len(dataset)):
        moment = dataset.get_moment(idx) / dataset.get_duration(idx)
        moment[1] -= 1e-6
        moment = (moment * num_chunks).long()
        frequency[moment[0], moment[1]] += 1

    logger = logging.getLogger("vmr.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    if result_output_dir:
        result_dict = {'vid': [], 'sentence': [], 'iou': [], 'pred_top5': [], 'duration': []}
    else:
        result_dict = None

    if predictions[0][0].shape[0]==1:
        recall_metrics=(1,)
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    
    # Initialization
    table = [['Rank@{},IoU@{:.1f}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    table_mtb = [['Rank@{},MTBIoU@{:.1f}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    recall_x_iou_unbiased = torch.zeros(num_chunks, num_chunks, num_recall_metrics, num_iou_metrics)
    recall_x_mtbiou = torch.zeros(num_recall_metrics, num_iou_metrics)
    recall_x_miou = torch.zeros(num_recall_metrics)

    # Calculation
    for idx in tqdm(range(len(predictions))):
        result = predictions[idx]
        duration = dataset.get_duration(idx)
        gt_moment = dataset.get_moment(idx)
        gt_chunk_indices = gt_moment / duration
        gt_chunk_indices[1] -= 1e-6
        gt_chunk_indices = (gt_chunk_indices * num_chunks).long()
        candidates, scores = result[0]*duration, result[1]

        predicted_moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)
        predicted_ious = iou(predicted_moments[:max(recall_metrics)], gt_moment)

        for i, r in enumerate(recall_metrics):
            ious_o = predicted_ious[:r] # [r/?]
            recall_x_miou[i] += ious_o.mean()
            if ious_o.size(0) < r:
                padding = r - ious_o.size(0)
                ious = torch.nn.functional.pad(ious_o, (0,padding), "constant", 0) # [r]
            else:
                ious = ious_o # [r]
            bools = ious[:,None].expand(r, num_iou_metrics) > iou_metrics # [r, num_iou_metrics]
            recall_x_iou[i] += bools.any(dim=0) # [num_iou_metrics]
            recall_x_iou_unbiased[gt_chunk_indices[0],gt_chunk_indices[1],i] += bools.any(dim=0)
            if r==1 and result_dict:
                result_dict['vid'].append(dataset.get_vid(idx))
                result_dict['sentence'].append(dataset.get_sentence(idx))
                result_dict['iou'].append(ious[0])
                result_dict['pred_top5'].append(predicted_moments)
                result_dict['duration'].append(duration)
    recall_x_iou /= len(predictions)
    recall_x_miou /= len(predictions)
    recall_x_iou_unbiased /= frequency[:,:,None,None]
    for i in range(len(recall_metrics)):
        for j in range(len(iou_metrics)):
            temp = recall_x_iou_unbiased[:,:,i,j]
            recall_x_mtbiou[i,j] = temp[~torch.isnan(temp)].mean()
    result_dict['mtbiou'] = recall_x_mtbiou
    result_dict['recall_x_iou_unbiased'] = recall_x_iou_unbiased
    result_dict['frequency'] = frequency
    
    # Print result in table
        # Original results
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
        # Mean of Temporal Boundary results
    table_mtb.append(['{:.02f}'.format(recall_x_mtbiou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table_mtb = AsciiTable(table_mtb)
    for i in range(num_recall_metrics*num_iou_metrics):
        table_mtb.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
    logger.info('\n' + table_mtb.table)
    
    # Print result in line
        # Original results
    result_line = ['Rank@{},IoU@{:.01f}={:.02f}'.format(recall_metrics[i],iou_metrics[j],recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)]
    result_line.extend(
        ['Rank@{}mIoU={:.02f}'.format(recall_metrics[i], recall_x_miou[i]*100) for i in range(num_recall_metrics)]
    )
    logger.info('\n' + '  '.join(result_line))
        # Mean of Temporal Boundary results
    result_line = ['Rank@{},MTBIoU@{:.01f}={:.02f}'.format(recall_metrics[i],iou_metrics[j],recall_x_mtbiou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)]
    logger.info('\n' + '  '.join(result_line))
    
    # Save results
    if result_output_dir:
        with open(result_output_dir + 'test_results.pkl', 'wb') as F: # DO NOT use join for prefix '/{k}_{epoch}e'
            pickle.dump(result_dict, F)
    
    return recall_x_iou[0,-1]


def evaluate_new(dataset_list, predictions_list, nms_thresh, result_output_dir, recall_metrics=(1,5), iou_metrics=(0.1,0.3,0.5,0.7)):
    """evaluate dataset using different methods based on dataset type.
        Args: 
            predictions: list[
                    (
                        moments_norm (tensor(num_predictions, 2)), 
                        scores (tensor(num_predictions))
                    )
                ]
        Returns:
            Recall@1 mIoU float
    """
    dataset_original, dataset_replaced = dataset_list
    predictions_original, predictions_replaced = predictions_list
    dataset_name = dataset_original.__class__.__name__

    logger = logging.getLogger("vmr.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset_original)))

    if result_output_dir:
        result_dict = {'vid': [], 'sentence': [], 'iou_o': [], 'iou_rp': []}
    else:
        result_dict = None

    if predictions_original[0][0].shape[0]==1:
        recall_metrics=(1,)
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    unbaised_recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    recall_x_miou = torch.zeros(num_recall_metrics)
    miss_x_iou_bias = torch.zeros(num_recall_metrics, num_iou_metrics) # number of samples failed to recall in replaced datasets

    for idx in tqdm(range(len(predictions_original))):
        duration = dataset_original.get_duration(idx)
        gt_moment = dataset_original.get_moment(idx)
        
        result_original = predictions_original[idx]
        candidates_original, scores_original = result_original[0]*duration, result_original[1]
        predicted_moments_original = nms(candidates_original, scores_original, topk=recall_metrics[-1], thresh=nms_thresh)
        predicted_ious_original = iou(predicted_moments_original[:max(recall_metrics)], gt_moment)
        
        result_replaced = predictions_replaced[idx]
        candidates_replaced, scores_replaced = result_replaced[0]*duration, result_replaced[1]
        predicted_moments_replaced = nms(candidates_replaced, scores_replaced, topk=recall_metrics[-1], thresh=nms_thresh)
        predicted_ious_replaced = iou(predicted_moments_replaced[:max(recall_metrics)], gt_moment)

        for i, r in enumerate(recall_metrics):
            ious_o = predicted_ious_original[:r] # [r/?]
            ious_rp = predicted_ious_replaced[:r] # [r/?]
            recall_x_miou[i] += ious_o.mean()
            
            ious_o = iou_padding(ious_o, r)
            ious_rp = iou_padding(ious_rp, r)
            bools_o = ious_o[:,None].expand(r, num_iou_metrics) > iou_metrics # [r, num_iou_metrics]
            bools_rp = ious_rp[:,None].expand(r, num_iou_metrics) < iou_metrics # [r, num_iou_metrics]
            unbaised_recall_x_iou[i] += bools_rp.all(dim=0)*bools_o.any(dim=0) # [num_iou_metrics]
            miss_x_iou_bias[i] += bools_rp.any(dim=0) # [num_iou_metrics]
            
            if i==1 and result_dict:
                result_dict['vid'].append(dataset_original.get_vid(idx))
                result_dict['sentence'].append(dataset_original.get_sentence(idx))
                result_dict['iou_o'].append(ious_o[:5])
                result_dict['iou_rp'].append(ious_rp[:5])
    
    hard_unbaised_recall_x_iou, unbaised_recall_x_iou = unbaised_recall_x_iou/miss_x_iou_bias, unbaised_recall_x_iou/len(predictions_original)
    recall_x_miou /= len(predictions_original)
    # Print result in table
    
    table = [['hard_UBRank@{},IoU@{:.1f}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    table.append(['{:.02f}'.format(hard_unbaised_recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
    
    table = [['UBRank@{},IoU@{:.1f}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    table.append(['{:.02f}'.format(unbaised_recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)

    if result_output_dir:
        with open(result_output_dir + 'test_results.pkl', 'wb') as F: # DO NOT use join for prefix '/{k}_{epoch}e'
            pickle.dump(result_dict, F)