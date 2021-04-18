import torch
from torch import nn
from torch.functional import F
import numpy as np
import copy

from .BUModules import HadamardProduct, ResBlock1D, NonLocalBlock, AttentivePooling
from .BUModules import TanhAttention, CrossGate, DynamicGRU
from .BUModules import CoAttention, CoAttention_intra, ConvGRUCell # CSMGAN
from ..Transformer import MultiHeadedAttention, AdaptiveSelfGatingAttention, PositionalEncoding, TransFormerLayer, CoTransFormerLayer, LayerNorm, SelfGating, clones
from ..Transformer import LightKeyCoTransFormerLayer, LightQueryCoTransFormerLayer


class LGIMMEncoder(nn.Module):
    def __init__(self, num_semantic_entity, query_hidden_dim, visual_hidden_dim, 
        mm_fusion_method, 
        l_type, resblock_kernel_size, num_local_blocks, do_downsample,
        g_type, num_attention, attention_use_embedding, num_global_blocks,
        num_nl_heads, nl_dropout, nl_use_bias, nl_use_local_mask
        ):
        super(LGIMMEncoder, self).__init__()
        self.nse = num_semantic_entity
        self.mm_fusion_method = mm_fusion_method
        self.l_type = l_type
        self.g_type = g_type

        # Multimodal fusion layer
        if mm_fusion_method == "mul":
            self.fusion_fn = self._make_modulelist(
                HadamardProduct(visual_hidden_dim, query_hidden_dim, visual_hidden_dim),
                self.nse
            )
        elif mm_fusion_method == "concat":
            self.lin_fn = self._make_modulelist(
                nn.Linear(2*visual_hidden_dim, visual_hidden_dim),
                self.nse
            )

        # Local interaction layer
        if l_type == "res_block":
            self.local_fn = self._make_modulelist(
                ResBlock1D(visual_hidden_dim, visual_hidden_dim, resblock_kernel_size, num_local_blocks, do_downsample),
                self.nse
            )
        elif l_type == "masked_nl":
            raise NotImplementedError
            nth_local_fn = self._make_modulelist(
                NonLocalBlock(config, "lgi_local"), n_local_mnl
            )
            self.local_fn = self._make_modulelist(nth_local_fn, self.nse)

        # Global interaction layer
        self.satt_fn = AttentivePooling(
            num_layer = num_attention,
            feat_dim = visual_hidden_dim,
            hidden_dim = visual_hidden_dim//2,
            use_embedding = attention_use_embedding,
            embedding_dim = visual_hidden_dim,
        )
        if g_type == "nl":
            self.n_global_nl = num_global_blocks
            self.global_fn = self._make_modulelist(
                NonLocalBlock(visual_hidden_dim, visual_hidden_dim, num_nl_heads, nl_dropout, nl_use_bias, nl_use_local_mask),
                self.n_global_nl
            )

    def forward(self, seg_feats, seg_masks, se_feats):
        """ Perform local-global video-text interactions
        1) modality fusion, 2) local context modeling, and 3) global context modeling
        Args:
            seg_feats: segment-level features; [B,seg,D]
            seg_masks: masks for effective segments in video; [B,seg]
            se_feats: semantic entity features; [B,N,D]
        Returns:
            sa_feats: semantic-aware segment features; [B,L,D]
        """
        if self.nse == 1:
            se_feats = se_feats.unsqueeze(1)
        assert self.nse == se_feats.size(1)
        B, nseg, _ = seg_feats.size()

        m_feats = self._segment_level_modality_fusion(seg_feats, se_feats)
        ss_feats = self._local_context_modeling(m_feats, seg_masks)
        sa_feats, sattw = self._global_context_modeling(ss_feats, se_feats, seg_masks)

        return sa_feats, sattw
    
    def _segment_level_modality_fusion(self, s_feats, se_feats):
        B, nseg, _ = s_feats.size()
        # fuse segment-level feature with individual semantic entitiey features
        m_feats = []
        for n in range(self.nse):
            q4s_feat = se_feats[:,n,:].unsqueeze(1).expand(B, nseg, -1)
            if self.mm_fusion_method == "concat":
                fused_feat = torch.cat([s_feats, q4s_feat], dim=2)
                fused_feat = torch.relu(self.lin_fn[n](fused_feat))
            elif self.mm_fusion_method == "add":
                fused_feat = s_feats + q4s_feat
            elif self.mm_fusion_method == "mul":
                fused_feat = self.fusion_fn[n]([s_feats, q4s_feat])
            else:
                raise NotImplementedError
            m_feats.append(fused_feat)

        return m_feats # N*[B*D]

    def _local_context_modeling(self, m_feats, masks):
        ss_feats = []

        for n in range(self.nse):
            if self.l_type == "res_block":
                l_feats = self.local_fn[n](m_feats[n].transpose(1,2)).transpose(1,2) # [B,nseg,*]
            elif self.l_type == "masked_nl":
                l_feats = m_feats[n]
                for s in range(self.n_local_mnl):
                    l_feats, _ = self.local_fn[n][s](l_feats, masks)
            else:
                raise NotImplementedError
                l_feats = feats
            ss_feats.append(l_feats)

        return ss_feats # N*[B,D]

    def _global_context_modeling(self, ss_feats, se_feats, seg_masks):
        ss_feats = torch.stack(ss_feats, dim=1) # N*[B,nseg,D] -> [B,N,nseg,D]

        # aggregating semantics-specific features
        _, sattw = self.satt_fn(se_feats)
        # [B,N,1,1] * [B,N,nseg,D] = [B,N,nseg,D]
        a_feats = sattw.unsqueeze(2).unsqueeze(2) * ss_feats
        a_feats = a_feats.sum(dim=1) # [B,nseg,D]

        # capturing contextual and temporal relations between semantic entities
        if self.g_type == "nl":
            sa_feats = a_feats
            for s in range(self.n_global_nl):
                sa_feats, _ = self.global_fn[s](sa_feats, seg_masks)
        else:
            sa_feats = a_feats

        return sa_feats, sattw
    
    def _make_modulelist(self, net, n):
        assert n > 0
        new_net_list = nn.ModuleList()
        new_net_list.append(net)
        if n > 1:
            for i in range(n-1):
                new_net_list.append(copy.deepcopy(net))
        return new_net_list


class CMINMMEncoder(nn.Module):
    def __init__(self, model_hidden_dim, video_segment_num, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.video_segment_num = video_segment_num
        self.v2s = TanhAttention(model_hidden_dim)
        self.cross_gate = CrossGate(model_hidden_dim)
        #self.bilinear = Bilinear(model_hidden_dim, model_hidden_dim, model_hidden_dim)
        self.rnn = DynamicGRU(model_hidden_dim << 1, model_hidden_dim >> 1, bidirectional=True, batch_first=True)

    def forward(self, segfeats, seglens, wordfeats, wordmasks):
        x1 = self.v2s(segfeats, wordfeats, wordmasks)
        frames1, x1 = self.cross_gate(segfeats, x1)
        mmfeats = torch.cat([frames1, x1], -1)
        # wordfeats = self.bilinear(frames1, x1, F.relu)
        mmfeats = self.rnn(mmfeats, seglens, self.video_segment_num)
        mmfeats = F.dropout(mmfeats, self.dropout, self.training)
        return mmfeats


class FIANMultiModalEncoder(nn.Module):
    def __init__(self, query_input_dim, max_num_words, visual_input_dim, num_segments, 
        feat_hidden_dim, feedforward_dim, num_heads, dropout):
        super(FIANMultiModalEncoder, self).__init__()
        self.max_num_words = max_num_words
        
        # CGA Module
        self.visual_CoTRM = CoTransFormerLayer(feat_hidden_dim, num_heads, feedforward_dim, dropout)
        self.textual_CoTRM = CoTransFormerLayer(feat_hidden_dim, num_heads, feedforward_dim, dropout)
        # Fusion Module
        self.textual_upsample1 = nn.Sequential(
            LayerNorm(feat_hidden_dim),
            nn.Conv1d(self.max_num_words, num_segments, 1)
        )
        self.textual_upsample2 = nn.Sequential(
            LayerNorm(feat_hidden_dim),
            nn.Conv1d(self.max_num_words, num_segments, 1)
        ) 
        self.self_gate = nn.Sequential(
             LayerNorm(feat_hidden_dim<<1),
            nn.Linear(feat_hidden_dim<<1, feat_hidden_dim<<1),
            nn.Sigmoid()
        ) 
        self.norm_mm = LayerNorm(feat_hidden_dim<<1)
        self.rnn = DynamicGRU(feat_hidden_dim<<1, feat_hidden_dim >> 1, bidirectional=True, batch_first=True)
    
    def forward(self, frame_embed, query_embed, seglens, wordmasks):
        """ encode query sequence
        Args:
            frame_embed [B, num_seg, vidim]
            seglens [B]
            query_embed (tensor[B, maxL, qidim])
            wordmasks (tensor[B, maxL])
        Returns:
            
        """
        B, num_seg, vhdim = frame_embed.size()
        maxL = query_embed.size(1)
        # Build Co-attention Mask
        b = torch.empty(B, num_seg, num_seg, dtype=torch.int32, device='cuda')
        for i in range(B):
            b[i] = torch.diag(F.pad(wordmasks[i], (0, num_seg-maxL)))
        b = b[:, :num_seg, :maxL]
        visual_query_mask = wordmasks.unsqueeze(1)|b
        query_visual_mask = wordmasks.unsqueeze(2).expand(B, maxL, num_seg)
        # CGA
        v_co_trm, q_co_trm = self.visual_CoTRM, self.textual_CoTRM
        query_embed1 = q_co_trm(query_embed, frame_embed, query_visual_mask)
        frame_embed1 = v_co_trm(frame_embed, query_embed1, visual_query_mask)
        frame_embed2 = v_co_trm(frame_embed, query_embed, visual_query_mask)
        query_embed2 = q_co_trm(query_embed, frame_embed2, query_visual_mask)
        # Fusion
        query_embed = torch.cat([self.textual_upsample1(query_embed1), self.textual_upsample2(query_embed2)], dim=-1)
        frame_embed = torch.cat([frame_embed1, frame_embed2], dim=-1)
        mm_embed = frame_embed + self.self_gate(query_embed)*query_embed
        mm_embed = self.rnn(self.norm_mm(mm_embed), seglens, num_seg)

        return {
            'frame_embed': mm_embed
        }

class CSMGANMultiModalEncoder(nn.Module):
    def __init__(self, model_hidden_dim, video_segment_num, max_num_words, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        #attentive graph
        self.atten = CoAttention(model_hidden_dim, model_hidden_dim)
        self.intra_v = CoAttention_intra(video_segment_num, model_hidden_dim)
        self.intra_s = CoAttention_intra(max_num_words, model_hidden_dim) 

        self.update_v = ConvGRUCell(model_hidden_dim, model_hidden_dim)
        self.update_s = ConvGRUCell(model_hidden_dim, model_hidden_dim)
        self.update_v_intra = ConvGRUCell(model_hidden_dim, model_hidden_dim)
        self.update_s_intra = ConvGRUCell(model_hidden_dim, model_hidden_dim)
        self.v2s = TanhAttention(model_hidden_dim)
        
        self.rnn = DynamicGRU(model_hidden_dim << 1, model_hidden_dim >> 1, bidirectional=True, batch_first=True)

    def forward(self, frames, seglens, x, node_mask):
        """
        frames [B, seg, vdim] segfeats
        seglens [B]
        x [B, len, wdim] wordfeats
        node_mask [B, len] wordmasks
        """
        frames_len = frames.shape[1]
        #attentive
        x1_att, x2_att, _, _ = self.atten(frames, x, node_mask)
        x1_m, x2_m = x1_att, x2_att#self.message_v(x1_att), self.message_s(x2_att)
        frames1 = self.update_v(x1_m, frames)
        x1 = self.update_s(x2_m, x)

        x1_m, _, a1, _ = self.intra_v(frames1, frames1, node_mask)
        x2_m, _, a2, _ = self.intra_s(x1, x1, node_mask)
        frames1 = self.update_v_intra(x1_m, frames1)
        x1 = self.update_s_intra(x2_m, x1)
        
        """
        Below is what exactly appeared in CSMGAN's offical code
        """
        #layer 2
        #x1_att, x2_att, a1, a2 = self.atten(frames1, x1, node_mask)
        #x1_m, x2_m = x1_att, x2_att#self.message_v(x1_att), self.message_s(x2_att)
        #frames1 = self.update_v(x1_m, frames1)
        #x1 = self.update_s(x2_m, x1)
        #x1_m, _, a1, _ = self.intra_v(frames1, frames1, node_mask)
        #x2_m, _, a2, _ = self.intra_s(x1, x1, node_mask)
        #frames1 = self.update_v_intra(x1_m, frames1)
        #x1 = self.update_s_intra(x2_m, x1)
        
        #frames1, x1 = frames, x
        #a1, a2 = 1, 1
        # interactive
        x1 = self.v2s(frames1, x1, node_mask)
        x = torch.cat([frames1, x1], -1) #x1
        x = self.rnn(x, seglens, frames_len)
        x = F.dropout(x, self.dropout, self.training)
        
        return x

def build_multimodal_encoder(cfg, arch):
    video_segment_num = cfg.INPUT.NUM_SEGMENTS
    max_num_words = cfg.INPUT.MAX_NUM_WORDS
    num_segments = cfg.INPUT.NUM_SEGMENTS
    if arch == 'LGI':
        num_semantic_entity = cfg.MODEL.LGI.QUERYENCODER.NUM_SEMANTIC_ENTITY
        query_hidden_dim = cfg.MODEL.LGI.QUERYENCODER.RNN.HIDDEN_SIZE
        visual_hidden_dim = cfg.MODEL.LGI.VIDEOENCODER.HIDDEN_SIZE

        mm_fusion_method = cfg.MODEL.LGI.MMENCODER.FUSION.METHOD

        l_type = cfg.MODEL.LGI.MMENCODER.LOCAL_CONTEXT.METHOD
        resblock_kernel_size = cfg.MODEL.LGI.MMENCODER.LOCAL_CONTEXT.RESBLOCK_KSIZE
        num_local_blocks = cfg.MODEL.LGI.MMENCODER.LOCAL_CONTEXT.RESBLOCK_NUM
        do_downsample = cfg.MODEL.LGI.MMENCODER.LOCAL_CONTEXT.DOWNSAMPLE

        g_type = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.METHOD
        num_attention = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.SATT_NUM
        attention_use_embedding = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.SATT_USE_EMBEDDING
        num_global_blocks = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.NL_NUM
        num_nl_heads = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.NL_HEADS_NUM
        nl_dropout = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.NL_DROPOUT
        nl_use_bias = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.NL_USE_BIAS
        nl_use_local_mask = cfg.MODEL.LGI.MMENCODER.GLOBAL_CONTEXT.NL_USE_LOCAL_MASK
        return LGIMMEncoder(
            num_semantic_entity, query_hidden_dim, visual_hidden_dim, 
            mm_fusion_method, 
            l_type, resblock_kernel_size, num_local_blocks, do_downsample,
            g_type, num_attention, attention_use_embedding, num_global_blocks,
            num_nl_heads, nl_dropout, nl_use_bias, nl_use_local_mask
        )
    elif arch == 'CMIN':
        model_hidden_dim = cfg.MODEL.CMIN.HIDDEN_DIM
        dropout = cfg.MODEL.CMIN.DROPOUT
        return CMINMMEncoder(
            model_hidden_dim,
            video_segment_num,
            dropout
        )
    elif arch == 'FIAN':
        query_input_dim = cfg.MODEL.FIAN.QUERYENCODER.HIDDEN_DIM
        visual_input_dim = cfg.MODEL.FIAN.VIDEOENCODER.HIDDEN_DIM
        feat_hidden_dim = cfg.MODEL.FIAN.MMENCODER.HIDDEN_DIM 
        feedforward_dim = cfg.MODEL.FIAN.MMENCODER.TRANSFORMER.FEEDFORWARD_DIM
        num_heads = cfg.MODEL.FIAN.MMENCODER.TRANSFORMER.NUM_HEADS
        dropout = cfg.MODEL.FIAN.MMENCODER.TRANSFORMER.DROPOUT
        return FIANMultiModalEncoder(
            query_input_dim, max_num_words, visual_input_dim, num_segments, feat_hidden_dim, 
            feedforward_dim, num_heads, dropout
        )
    elif arch == 'CSMGAN':
        model_hidden_dim = cfg.MODEL.CMIN.HIDDEN_DIM
        dropout = cfg.MODEL.CMIN.DROPOUT
        return CSMGANMultiModalEncoder(
            model_hidden_dim,
            video_segment_num,
            max_num_words,
            dropout
        )
    else:
        raise NotImplementedError