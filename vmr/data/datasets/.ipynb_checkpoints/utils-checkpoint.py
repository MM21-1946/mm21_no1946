import os
from os.path import join, exists
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import h5py
import numpy as np

import torch
import torchtext
import torch.nn.functional as F

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s) + 1
    union = end.max(e) - start.min(s) + 1
    return inter.clamp(min=0) / union

def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = torch.nonzero(score2d)   
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d

def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

def generate_anchorbased_proposals(widths, num_segments, moment, duration, threshold=0.3):
    """
    args:
        widths: list(int)
        num_segments: int
        moment: torch.tensor(2,)
        duration: float
        threshold: float
    """

    """
    Warning! This anchor generation method is from CMIN, which assume that all videos
    are sampled into the same length!
    """

    widths = np.array(widths)
    label = np.array([int(moment[0]/duration*num_segments), int(0.5+moment[1]/duration*num_segments)])

    center = 7.5
    start = center - 0.5 * (widths - 1)
    end = center + 0.5 * (widths - 1)
    anchors = np.stack([start, end], -1)
    widths = (anchors[:, 1] - anchors[:, 0] + 1)  # [num_anchors]
    centers = np.arange(0, num_segments)  # [video_len]
    start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
    end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
    proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

    proposals = np.reshape(proposals, [-1, 2]) # [video_len* num_anchors, 2]
    illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= num_segments)
    label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
    IoUs = calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                        (label1[:, 0], label1[:, 1]))
    IoUs[illegal] = 0.0  # [video_len * num_anchors]
    max_IoU = np.max(IoUs)
    assert max_IoU > 0.0
    IoUs[IoUs < threshold * max_IoU] = 0.0
    IoUs = IoUs / max_IoU
    scores = IoUs.astype(np.float32)
    scores_mask = (1 - illegal).astype(np.uint8)
    return torch.from_numpy(scores), torch.from_numpy(scores_mask)


def generate_slidingwindow_proposals(widths, num_segments, moment, duration, overlap):
    """
    args:
        widths: list(int)
        num_segments: int
        moment: torch.tensor(2,)
        duration: float
        threshold: float
    """

    """
    Warning! This anchor generation method is from FIAN, which assume that all videos
    are sampled into the same length!
    """
    widths = np.array(widths)
    overlap = np.array(overlap)
    label = np.array([int(moment[0]/duration*num_segments), int(0.5+moment[1]/duration*num_segments)])
    
    strides = (widths*(1-overlap)).astype(int)
    prop_lens = ((num_segments-widths)/strides+1).astype(int)
    proposal = []
    for prop_len, width, stride in zip(prop_lens, widths, strides):
        for k in range(prop_len):
            proposal.append([k*stride, k*stride+width-1])
    proposals = np.array(proposal) # [num_proposals, 2]
    label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
    IoUs = calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                        (label1[:, 0], label1[:, 1]))
    max_IoU = np.max(IoUs)
    IoUs = IoUs / max_IoU
    scores = IoUs.astype(np.float32)
    return torch.from_numpy(scores)


def get_fixed_length_feats(feats, num_pre_clips, upsample):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    if num_src_clips <= num_pre_clips and not upsample:
        vid_feats = feats.new_zeros((num_pre_clips, feats.size(1)))
        vid_feats[:num_src_clips] = feats
        return vid_feats, num_src_clips
    else:
        idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
        idxs = idxs.round().long().clamp(max=num_src_clips-1)
        # To prevent a empty selection, check the idxs
        meanfeats = []
        for i in range(num_pre_clips):
            s, e = idxs[i], idxs[i+1]
            if s < e:
                meanfeats.append(feats[s:e].mean(dim=0))
            else:
                meanfeats.append(feats[s])
    return torch.stack(meanfeats), num_pre_clips
    
def video2feats(feat_file, vids, num_pre_clips, dataset_name, upsample):
    assert exists(feat_file)
    vid_feats = {}
    vid_seglen = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid][:]
            elif dataset_name == "tacos":
                feat = f[vid][:]
            elif dataset_name == "charades":
                feat = f[vid][:].squeeze()
            else:
                raise NotImplementedError
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            vid_feats[vid], vid_seglen[vid] = get_fixed_length_feats(feat, num_pre_clips, upsample)
    return vid_feats, vid_seglen

def glove_embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat(
            [vocab.vectors, torch.zeros(1, vocab.dim)],
            dim=0
        )
        vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) \
        for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)
