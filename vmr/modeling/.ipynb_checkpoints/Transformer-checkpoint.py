import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, group_prob=None, no_cuda=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        q_len = query.size()[-2]
        kv_len = value.size()[-2]
        if mask.size(2) == 1:
            # Self-attention
            if no_cuda:
                b = torch.from_numpy(np.diag(np.ones(kv_len, dtype=np.int32),0))
            else:
                b = torch.from_numpy(np.diag(np.ones(kv_len, dtype=np.int32), 0)).cuda()
            scores = scores.masked_fill((mask|b) == 0, -1e9)
        else:
            # Co-attention
            assert mask.size(2) == q_len
            scores = scores.masked_fill((mask) == 0, -1e9)
    if group_prob is not None:
        p_attn = F.softmax(scores, dim = -1)
        p_attn = p_attn*group_prob.unsqueeze(1)  # (2)
    else:
        p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, no_cuda=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_cuda = no_cuda
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob, no_cuda=self.no_cuda)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class AdaptiveSelfGatingAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, no_cuda=False):
        "Take in model size and number of heads."
        super(AdaptiveSelfGatingAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.gatefc_q = nn.Linear(self.d_k, self.d_k)
        self.gatefc_k = nn.Linear(self.d_k, self.d_k)
        self.gatefc_g = nn.Linear(self.d_k, self.d_k*2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_cuda = no_cuda
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Self Gating
        G = self.gatefc_q(query) * self.gatefc_k(key)
        M = F.sigmoid(self.gatefc_g(G)) # (bs, h, num_region, d_k*2)
        query = query * M[:, :, :, :self.d_k]
        key = key * M[:, :, :, self.d_k:]
        
        # 3) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob, no_cuda=self.no_cuda)
        
        # 4) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8, no_cuda=False):
        super(GroupAttention, self).__init__()
        self.d_model = 256.
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        #self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_cuda = no_cuda

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        context =self.norm(context)

        if self.no_cuda:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1))
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0))
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1))
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0))
        else:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),1)).cuda()
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32),0)).cuda()
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32),-1)).cuda()
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len,seq_len], dtype=np.float32),0)).cuda()

        #mask = eos_mask & (a+c) | b
        mask = eos_mask & (a+c)
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        
        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-9)  #(7)
        neibor_attn = prior + (1. - prior)*neibor_attn  # (8)

        t = torch.log(neibor_attn + 1e-9).masked_fill(a==0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int()-b)==0, 0)     
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b==0, 1e-9)
        
        return g_attn, neibor_attn


class PositionwiseFeedForward(nn.Module):
    "Implements Transformer FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.w_2(self.dropout(gelu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    
class SelfGating(nn.Module):
    def __init__(self, size, dropout):
        super(SelfGating, self).__init__()
        self.proj = nn.Linear(size, size)

    def forward(self, x):
        return x * torch.sigmoid(self.proj(x))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class SelfGatedSublayerConnection(nn.Module):
    """
    A self-gated residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SelfGatedSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        #self.proj_base = nn.Linear(size, size)
        self.proj_residual = nn.Linear(size, size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #x = self.proj_base(x) + self.proj_residual(self.dropout(sublayer(self.norm(x))))
        x = x + self.dropout(sublayer(self.norm(x)))
        return torch.sigmoid(self.proj_residual(x)) * x


class TransFormerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hiddem_dim, num_heads, feedforward_dim, dropout, group_attn=False):
        super(TransFormerLayer, self).__init__()
        self.hiddem_dim = hiddem_dim
        
        self.self_attn = MultiHeadedAttention(num_heads, hiddem_dim)
        self.feed_forward = PositionwiseFeedForward(hiddem_dim, feedforward_dim, dropout)
        self.group_attn = GroupAttention(hiddem_dim) if group_attn else None
        self.sublayer = clones(SublayerConnection(hiddem_dim, dropout), 2)

    def forward(self, x, mask, group_prob):
        if self.group_attn:
            group_prob, break_prob = self.group_attn(x, mask, group_prob)
        else:
            break_prob = None
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob


class CoTransFormerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hiddem_dim, num_heads, feedforward_dim, dropout):
        super(CoTransFormerLayer, self).__init__()
        self.hiddem_dim = hiddem_dim
        
        self.co_attn = MultiHeadedAttention(num_heads, hiddem_dim)
        self.feed_forward = PositionwiseFeedForward(hiddem_dim, feedforward_dim, dropout)
        self.sublayer = nn.ModuleList([
            SelfGatedSublayerConnection(hiddem_dim, dropout),
            SublayerConnection(hiddem_dim, dropout)
        ])

    def forward(self, x, y, mask):
        group_prob = None
        x = self.sublayer[0](x, lambda x: self.co_attn(x, y, y, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward)


class LightKeyCoTransFormerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hiddem_dim, num_heads, feedforward_dim, dropout):
        super(LightKeyCoTransFormerLayer, self).__init__()
        self.hiddem_dim = hiddem_dim
        
        self.co_attn = LightKeyMultiHeadedAttention(num_heads, hiddem_dim)
        self.feed_forward = PositionwiseFeedForward(hiddem_dim, feedforward_dim, dropout)
        self.sublayer = nn.ModuleList([
            SelfGatedSublayerConnection(hiddem_dim, dropout),
            SublayerConnection(hiddem_dim, dropout)
        ])

    def forward(self, x, y, mask):
        group_prob = None
        x = self.sublayer[0](x, lambda x: self.co_attn(x, y, y, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward)

    
class LightQueryCoTransFormerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hiddem_dim, num_heads, feedforward_dim, dropout):
        super(LightQueryCoTransFormerLayer, self).__init__()
        self.hiddem_dim = hiddem_dim
        
        self.co_attn = LightQueryMultiHeadedAttention(num_heads, hiddem_dim)
        self.feed_forward = PositionwiseFeedForward(hiddem_dim, feedforward_dim, dropout)
        self.sublayer = nn.ModuleList([
            SelfGatedSublayerConnection(hiddem_dim, dropout),
            SublayerConnection(hiddem_dim, dropout)
        ])

    def forward(self, x, y, mask):
        group_prob = None
        x = self.sublayer[0](x, lambda x: self.co_attn(x, y, y, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward)


class LightKeyMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, no_cuda=False):
        "Take in model size and number of heads."
        super(LightKeyMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_cuda = no_cuda
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        """
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        """
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob, no_cuda=self.no_cuda)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #return self.linears[-1](x)
        return x

class LightQueryMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, no_cuda=False):
        "Take in model size and number of heads."
        super(LightQueryMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        #self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_cuda = no_cuda
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        """
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        """
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[0](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[1](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob, no_cuda=self.no_cuda)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        #return x