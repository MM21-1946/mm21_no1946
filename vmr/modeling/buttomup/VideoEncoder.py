import torch
from torch import nn
from torch.functional import F

from .BUModules import MultiHeadAttention, DynamicGRU
from ..Transformer import PositionalEncoding, MultiHeadedAttention, LayerNorm, clones


class LGIVideoEncoder(nn.Module):
    def __init__(self, visual_input_dim, visual_hidden_dim, position_encoding, video_segment_num=None):
        super(LGIVideoEncoder, self).__init__()
        self.position_encoding = position_encoding
        self.vid_emb_fn = nn.Sequential(*[
            nn.Linear(visual_input_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])
        if position_encoding:
            self.pos_emb_fn = nn.Sequential(*[
                nn.Embedding(video_segment_num, visual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])

    def forward(self, seg_feats, seg_masks):
        """ encode query sequence
        Args:
            seg_feats (tensor[B, seg, visual_indim])
            seg_masks: mask for effective segments; [B,seg]
        Returns:
            seg_emb (tensor[B, seg, visual_hdim])
        """
        seg_emb = self.vid_emb_fn(seg_feats) * seg_masks.float().unsqueeze(2)
        if self.position_encoding:
            """
            Note 12.3.2020
                Below is the LGI original positional embedding, it's wrong for the videos whose segment number are 
            bigger than NUM_SEGMENTS since they are downsampled, the 'pos' variable should be mulplyied by a 
            scaling factor.
                It maynot cause too much trouble in paper because in charades-STA datasets such phenomenon is rare.
            """
            # use absolute position embedding
            pos = torch.arange(0, seg_masks.size(1)).type_as(seg_masks).unsqueeze(0).long()
            pos_emb = self.pos_emb_fn(pos)
            B, nseg, pdim = pos_emb.size()
            pos_feats = (pos_emb.expand(B, nseg, pdim) * seg_masks.unsqueeze(2).float())
            seg_emb += pos_feats
        return seg_emb


class CMINVideoEncoder(nn.Module):
    def __init__(self, visual_input_dim, video_segment_num, model_hidden_dim, num_heads=8, num_attn_layers=2, attn_width=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.video_segment_num = video_segment_num
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(visual_input_dim, num_heads)
            for _ in range(num_attn_layers)
        ])
        self.rnn = DynamicGRU(visual_input_dim, model_hidden_dim >> 1, bidirectional=True, batch_first=False)
        self.self_attn_mask = torch.empty(self.video_segment_num, self.video_segment_num) \
            .float().fill_(float(-1e10)).cuda()
        for i in range(0, self.video_segment_num):
            low = i - attn_width
            low = 0 if low < 0 else low
            high = i + attn_width + 1
            high = self.video_segment_num if high > self.video_segment_num else high
            # attn_mask[i, low:high] = 0
            self.self_attn_mask[i, low:high] = 0
    
    def forward(self, seg_feats, seglen):
        """
            seg_feats (tensor[B, seg, feat_dim])
            seglen (tensor[B])
        """
        seg_feats = F.dropout(seg_feats, self.dropout, self.training)
        seg_feats = seg_feats.transpose(0, 1)

        for attention in self.attn_layers:
            res = seg_feats
            seg_feats, _ = attention(seg_feats, seg_feats, seg_feats, None, attn_mask=self.self_attn_mask)
            seg_feats = F.dropout(seg_feats, self.dropout, self.training)
            seg_feats = res + seg_feats

        seg_feats = self.rnn(seg_feats, seglen, self.video_segment_num)
        seg_feats = F.dropout(seg_feats, self.dropout, self.training)

        seg_feats = seg_feats.transpose(0, 1)
        return seg_feats
        

def build_video_encoder(cfg, arch):
    visual_input_dim = cfg.DATASETS.VISUAL_DIM
    video_segment_num = cfg.INPUT.NUM_SEGMENTS
    if arch == 'LGI':
        visual_hidden_dim = cfg.MODEL.LGI.VIDEOENCODER.HIDDEN_SIZE
        position_encoding = cfg.MODEL.LGI.VIDEOENCODER.USE_POSITION
        return LGIVideoEncoder(
            visual_input_dim, visual_hidden_dim, position_encoding, video_segment_num
        )
    elif arch == 'CMIN':
        model_hidden_dim = cfg.MODEL.CMIN.HIDDEN_DIM
        num_heads = cfg.MODEL.CMIN.VIDEOENCODER.NUM_HEADS
        num_attn_layers = cfg.MODEL.CMIN.VIDEOENCODER.NUM_ATT_LAYERS
        attn_width = cfg.MODEL.CMIN.VIDEOENCODER.ATT_WIDTH
        dropout = cfg.MODEL.CMIN.DROPOUT
        return CMINVideoEncoder(
            visual_input_dim, video_segment_num, model_hidden_dim, num_heads, num_attn_layers, attn_width, dropout
        )
    elif arch == 'FIAN':
        model_hidden_dim = cfg.MODEL.FIAN.VIDEOENCODER.HIDDEN_DIM
        num_attn_layers = cfg.MODEL.FIAN.VIDEOENCODER.NUM_ATT_LAYERS
        num_heads = cfg.MODEL.FIAN.VIDEOENCODER.NUM_HEADS
        attn_width = cfg.MODEL.FIAN.VIDEOENCODER.ATT_WIDTH
        dropout = cfg.MODEL.FIAN.VIDEOENCODER.DROPOUT
        return CMINVideoEncoder(
            visual_input_dim, video_segment_num, model_hidden_dim, num_heads, num_attn_layers, attn_width, dropout
        )
    elif arch == 'CSMGAN':
        model_hidden_dim = cfg.MODEL.CSMGAN.HIDDEN_DIM
        num_heads = cfg.MODEL.CSMGAN.VIDEOENCODER.NUM_HEADS
        num_attn_layers = cfg.MODEL.CSMGAN.VIDEOENCODER.NUM_ATT_LAYERS
        attn_width = cfg.MODEL.CSMGAN.VIDEOENCODER.ATT_WIDTH
        dropout = cfg.MODEL.CSMGAN.DROPOUT
        return CMINVideoEncoder(
            visual_input_dim, video_segment_num, model_hidden_dim, num_heads, num_attn_layers, attn_width, dropout
        )
    else:
        raise NotImplementedError