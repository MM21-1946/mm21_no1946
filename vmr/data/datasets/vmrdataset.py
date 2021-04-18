import re
import os
import copy
from os.path import join, dirname
import json
import pickle
import random
import logging
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForPreTraining

from .utils import video2feats, moment_to_iou2d, glove_embedding
from .utils import generate_anchorbased_proposals, generate_slidingwindow_proposals


class VMRDataset(torch.utils.data.Dataset):
    """
    num_segments: number of sampled feature vectors in a video
    upsample: upsample if number of feature vectors in a given video smaller than num_segments, otherwise padding
    in_memory: if visual features are loaded into RAM at the beginning of training

    For new proposal method, just define your own __generate_ground_truth__() function
    """
    def __init__(self, **kwargs):
        super(VMRDataset, self).__init__()
        
        self.anno_file = kwargs['anno_file']
        self.max_num_words = kwargs['max_num_words']
        self.fix_num_words = kwargs['fix_num_words']
        self.pre_query_size = kwargs['pre_query_size']
        self.dep_graph = kwargs['dep_graph']
        self.consti_mask = kwargs['consti_mask']
        self.tree_depth = kwargs['tree_depth']
        self.word2vec = kwargs['word2vec']
        self.dataset_name = kwargs['dataset_name']
        self.training = kwargs['training']
        self.feat_file = kwargs['feat_file']
        self.num_segments = kwargs['num_segments']
        self.upsample = kwargs['upsample']
        self.in_memory = kwargs['in_memory']
        self.resample = kwargs['resample']
        self.resample_weight = kwargs['resample_weight']
        
        self.slot = random.Random()
        self.avg_wordvec = torch.zeros(self.pre_query_size)
        
        self.parsing_suffix(kwargs['suffix'])
        self.load_annotations(**kwargs)
        self.weights = self.generate_weights() if self.resample else None
    
    def parsing_suffix(self, suffix):
        logger = logging.getLogger("vmr.trainer")
        dataset = 'train' if self.training else 'test'
        
        qmask_ratio = re.search(r'qm\d\d', suffix)
        qshuffle_ratio = re.search(r'qs\d\d', suffix)
        vmask_ratio = re.search(r'vm\d\d', suffix)
        vshuffle_ratio = re.search(r'vs\d\d', suffix)
        
        if qmask_ratio:
            qmask_ratio = qmask_ratio.string
            qmask_ratio = int(qmask_ratio[2]) + int(qmask_ratio[3])*0.1
            assert qmask_ratio >= 0. and qmask_ratio<=1.
            logger.info(f'>>>Set mask ratio of this {dataset} dataset to {qmask_ratio}<<<')
        if qshuffle_ratio:
            qshuffle_ratio = qshuffle_ratio.string
            qshuffle_ratio = int(qshuffle_ratio[2]) + int(qshuffle_ratio[3])*0.1
            assert qshuffle_ratio >= 0. and qshuffle_ratio<=1.
            logger.info(f'>>>Set shuffle ratio of this {dataset} dataset to {qshuffle_ratio}<<<')
        if vmask_ratio:
            vmask_ratio = vmask_ratio.string
            vmask_ratio = int(vmask_ratio[2]) + int(vmask_ratio[3])*0.1
            assert vmask_ratio >= 0. and vmask_ratio<=1.
            logger.info(f'>>>Set mask ratio of this {dataset} dataset to {vmask_ratio}<<<')
        if vshuffle_ratio:
            vshuffle_ratio = vshuffle_ratio.string
            vshuffle_ratio = int(vshuffle_ratio[2]) + int(vshuffle_ratio[3])*0.1
            assert vshuffle_ratio >= 0. and vshuffle_ratio<=1.
            logger.info(f'>>>Set shuffle ratio of this {dataset} dataset to {vshuffle_ratio}<<<')
        self.qmask_ratio = qmask_ratio
        self.qshuffle_ratio = qshuffle_ratio
        self.vmask_ratio = vmask_ratio
        self.vshuffle_ratio = vshuffle_ratio
    
    def load_annotations(self, proposal_method, **kwargs):
        logger = logging.getLogger("vmr.trainer")
        logger.info("Preparing data form file {}, please wait...".format(self.anno_file))
        self.annos = []
        self.gts = []
        word2vec_cache_prefix = os.path.splitext(self.anno_file)[0]
        word2vec_cache_file = '{}_word2vec_{}.pkl'.format(word2vec_cache_prefix, self.word2vec)

        # Define word embedding function
        if os.path.exists(word2vec_cache_file):
            annos_original = None
            # Load word embeddings cache if exists
            logger.info("Word2vec cache exist, load cache file.")
            with open(word2vec_cache_file, 'rb') as F:
                self.annos_query = pickle.load(F)
            def word_embedding(idx, sentence):
                assert self.annos_query[idx]['sentence'] == sentence, \
                    'annotation file {} has been modified, cache file expired!'.format(self.anno_file,)
                return self.annos_query[idx]['query'], self.annos_query[idx]['wordlen']
        else:
            annos_original = []
            # Computing word embeddings if there's no cache
            if self.word2vec == 'BERT':
                # Here we use second-to-last hidden layer
                # See 3.5 Pooling Strategy & Layer Choice in https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                bert = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)
                bert.to('cuda')
                def word_embedding(idx, sentence):
                    sentence_tokenized = tokenizer(sentence, return_tensors="pt") # token_num = sentence_num+2
                    for key in sentence_tokenized:
                        sentence_tokenized[key] = sentence_tokenized[key].to('cuda')
                    with torch.no_grad():
                        query = bert(**sentence_tokenized, output_hidden_states=True)['hidden_states'][-2].squeeze_().to('cpu') #(token_num, 768)
                        query = query[1:-1]
                    return query, query.size(0) #(sentence_len, 768) including punctuations
            elif self.word2vec == 'GloVe':
                def word_embedding(idx, sentence):
                    word2vec = glove_embedding(sentence)
                    return word2vec, word2vec.size(0) #(sentence_len, 300) including punctuations
            else:
                raise NotImplementedError
        
        # Loading annotations, generate ground truth for model proposal
        logger.info("loading annotations ...")
        with open(self.anno_file, 'r') as f:
            annos = json.load(f)
        for vid, anno in tqdm(annos.items()):
            duration = anno['duration'] if self.dataset_name != 'tacos' else anno['num_frames']/anno['fps']
            # Produce annotations
            for idx in range(len(anno['timestamps'])):
                timestamp = anno['timestamps'][idx]
                sentence = anno['sentences'][idx]
                if timestamp[0] < timestamp[1]:
                    moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)]) if self.dataset_name != 'tacos' \
                    else torch.tensor(
                            [max(timestamp[0]/anno['fps'],0), 
                            min(timestamp[1]/anno['fps'],duration)]
                        )
                    query, wordlen = word_embedding(len(self.annos), sentence)
                    self.avg_wordvec += query.mean(dim=0)
                    if annos_original is not None:
                        annos_original.append({
                            'sentence': sentence,
                            'query': query,
                            'wordlen': wordlen,
                        })
                    adjmat = torch.tensor(anno['dependency_parsing_graph'][idx]) if self.dep_graph else None
                    if self.consti_mask:
                        constimask = torch.tensor(anno['constituency_parsing_mask'][idx], dtype=torch.float32)
                        layers = torch.linspace(constimask.size(0)-1, 0, self.tree_depth).long() # The original tree is from root to leaf
                        constimask = constimask[layers,:,:]
                    else:
                        constimask = None
                    if self.dep_graph:
                        padding = query.size(0) - adjmat.size(0)
                    adjmat = torch.nn.functional.pad(adjmat, (0,padding,0,padding), "constant", 0) if self.dep_graph else None
                    if wordlen >= self.max_num_words:
                        wordlen = self.max_num_words
                        query = query[:self.max_num_words]
                        adjmat = adjmat[:self.max_num_words, :self.max_num_words] if self.dep_graph else None
                    elif self.fix_num_words:
                        padding = self.max_num_words - wordlen
                        query = torch.nn.functional.pad(query, (0,0,0,padding), "constant", 0)
                        #print('padded:', query.shape)
                        if self.dep_graph:
                            padding = self.max_num_words - adjmat.size(0)
                        adjmat = torch.nn.functional.pad(adjmat, (0,padding,0,padding), "constant", 0) if self.dep_graph else None
                    
                    self.annos.append(
                        {
                            'vid': vid,
                            'moment': moment,
                            'sentence': sentence,
                            'query': query,
                            'querymask': torch.ones(wordlen, dtype=torch.int32),
                            'adjmat': adjmat,
                            'constimask': constimask,
                            'wordlen': wordlen,
                            'duration': duration,
                        }
                    )
                    gt_dict = self.__generate_ground_truth__(moment, duration, proposal_method, **kwargs)
                    self.gts.append(gt_dict)
        
        self.avg_wordvec /= len(self.annos)
        
        if not os.path.exists(word2vec_cache_file):
            with open(word2vec_cache_file, 'wb') as F:
                word2vec_cache = [{'sentence': anno['sentence'], 'query': anno['query'], 'wordlen': anno['wordlen']} for anno in annos_original]
                pickle.dump(word2vec_cache, F)
            
        # Loading visual features if in_memory
        if self.in_memory:
            logger.info("Loading visual features from {}, please wait...".format(self.feat_file))
            self.feats, self.seglens = video2feats(self.feat_file, annos.keys(), self.num_segments, self.dataset_name, self.upsample)   
        logger.info("Dataset prepared!")
    
    def generate_weights(self):
        # 2d grid
        num_clips = 1000
        frequency = torch.zeros((num_clips, num_clips))
        for anno in self.annos:
            moment_norm = anno['moment'] / anno['duration']
            moment_norm[1] -= 1e-4
            moment_norm = (moment_norm * num_clips).long()
            frequency[moment_norm[0], moment_norm[1]] += 1
        class_prob = torch.pow(frequency, self.resample_weight)
        class_prob[frequency==0] = 0
        class_prob = class_prob / class_prob.sum()
        class_weights = class_prob / frequency
        weights = []
        for anno in self.annos:
            moment_norm = anno['moment'] / anno['duration']
            moment_norm[1] -= 1e-4
            moment_norm = (moment_norm * num_clips).long()
            weights.append(class_weights[moment_norm[0], moment_norm[1]].item())
        """
        # Head middle tail
        frequency = torch.zeros(3) # head middle tail
        thr = 1./16.
        for anno in self.annos:
            moment_norm = anno['moment'] / anno['duration']
            if moment_norm[0] < thr:
                frequency[0] += 1
            elif moment_norm[1] < 1.-thr: # middle time
                frequency[1] += 1
            else:
                frequency[2] += 1
        class_prob = torch.pow(frequency, self.resample_weight)
        class_prob = class_prob / class_prob.sum()
        class_weights = (class_prob / frequency).numpy()
        self.weights = []
        for anno in self.annos:
            moment_norm = anno['moment'] / anno['duration']
            if moment_norm[0] < thr:
                self.weights.append(class_weights[0])
            elif moment_norm[1] < 1.-thr: # middle time
                self.weights.append(class_weights[1])
            else:
                self.weights.append(class_weights[2])
        """
        return weights
    
    def __generate_ground_truth__(self, moment, duration, proposal_method, **kwargs):
        gt_dict = {}
        if proposal_method == 'standard':
            gt_dict['start_pos_norm'] = moment[0]/duration
            gt_dict['end_pos_norm'] = moment[1]/duration
        elif proposal_method == '2dtan':
            assert 'num_clips' in kwargs, \
                "For proposal method '2dtan', you must define num_clips!"
            gt_dict['iou2d'] = moment_to_iou2d(moment, kwargs['num_clips'], duration) # [num_clips, num_clips]
        elif proposal_method == 'anchor':
            assert 'anchor_widths' in kwargs, \
                "For proposal method 'anchor', you must define anchor_widths!"
            assert type(kwargs['anchor_widths']) is list, \
                "'anchor_widths' must be an int list!"
            assert self.upsample, \
                "Current anchor generation method only support fixed length video!"
            gt_dict['start_pos_norm'] = moment[0]/duration
            gt_dict['end_pos_norm'] = moment[1]/duration
            gt_dict['iou1d'], gt_dict['iou1d_mask'] = generate_anchorbased_proposals(
                kwargs['anchor_widths'], self.num_segments, moment, duration, threshold=0.3
            ) # clearing thershold = 0.3 for cmin
        elif proposal_method == 'sliding_window':
            assert 'window_widths' in kwargs, \
                "For proposal method 'sliding_window', you must define window_widths!"
            assert 'window_overlap' in kwargs, \
                "For proposal method 'sliding_window', you must define window_overlap!"
            assert self.upsample, \
                "Current sliding_window generation method only support fixed length video!"
            gt_dict['start_pos_norm'] = moment[0]/duration
            gt_dict['end_pos_norm'] = moment[1]/duration
            gt_dict['iou1d'] = generate_slidingwindow_proposals(
                kwargs['window_widths'], self.num_segments, moment, duration, kwargs['window_overlap']
            )
        else:
            raise NotImplementedError
        return gt_dict

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        if self.in_memory:
            feat, seglen = self.feats[vid], self.seglens[vid]
        else:
            feats, seglens = video2feats(self.feat_file, [vid], self.num_segments, self.dataset_name, self.upsample)
            feat, seglen = feats[vid], seglens[vid]
        
        # Mask and shuffle
        wordlen = anno['wordlen']
        query = anno['query']
        adjmat = anno['adjmat']
        constimask = anno['constimask']
        querymask = anno['querymask']
        
        #sentence_o = anno['sentence'].split(' ') #
        # Query attack
        if self.qmask_ratio:
            if not self.training:
                self.slot.seed(idx)
            masked_indices = self.slot.sample(range(wordlen), int(self.qmask_ratio*wordlen))
            query[masked_indices] = torch.zeros(self.pre_query_size)
            #sentence_o = ['?' if idx in masked_indices else word for idx, word in enumerate(sentence_o)] # 
        if self.qshuffle_ratio:
            if not self.training:
                self.slot.seed(idx)
            to_shuffle_indices = self.slot.sample(range(wordlen), int(self.qshuffle_ratio*wordlen))
            shuffled_indices = copy.copy(to_shuffle_indices)
            self.slot.shuffle(shuffled_indices)
            query[to_shuffle_indices] = query[shuffled_indices]
            #sentence = copy(sentence_o) #
            #for idx in range(len(sentence_o)): #
            #    sentence[to_shuffle_indices[idx]] = sentence_o[shuffled_indices[idx]] #
            if self.dep_graph:
                adjmat[to_shuffle_indices, to_shuffle_indices] = adjmat[shuffled_indices, shuffled_indices]
            if self.consti_mask:
                constimask[:, to_shuffle_indices, to_shuffle_indices] = constimask[:, shuffled_indices, shuffled_indices]
        # Visual attack
        if self.vmask_ratio:
            if not self.training:
                self.slot.seed(idx)
            masked_indices = self.slot.sample(range(seglen), int(self.vmask_ratio*seglen))
            feat[masked_indices] = torch.zeros(feat.size(1), dtype=feat.dtype)
        if self.vshuffle_ratio:
            if not self.training:
                self.slot.seed(idx)
            to_shuffle_indices = self.slot.sample(range(seglen), int(self.vshuffle_ratio*seglen))
            shuffled_indices = copy.copy(to_shuffle_indices)
            self.slot.shuffle(shuffled_indices)
            feat[to_shuffle_indices] = feat[shuffled_indices]
        return idx, feat, seglen, query, querymask, adjmat, constimask, wordlen, self.gts[idx]
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']
