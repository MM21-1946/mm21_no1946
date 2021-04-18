import math
import copy

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.autograd import Variable


class HadamardProduct(nn.Module):
    def __init__(self, idim_1, idim_2, hdim):
        super(HadamardProduct, self).__init__() # Must call super __init__()

        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, inp):
        """
        Args:
            inp1: [B,idim_1] or [B,L,idim_1]
            inp2: [B,idim_2] or [B,L,idim_2]
        """
        x1, x2 = inp[0], inp[1]
        return torch.relu(self.fc_3(torch.relu(self.fc_1(x1)) * torch.relu(self.fc_2(x2))))


class ResBlock1D(nn.Module):
    def __init__(self, idim, odim, ksize, nblocks, downsample=False):
        super(ResBlock1D, self).__init__() # Must call super __init__()

        # get configuration
        self.nblocks = nblocks
        self.do_downsample = downsample

        # set layers
        if self.do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(idim, odim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(odim),
            )
        self.blocks = nn.ModuleList()
        for i in range(self.nblocks):
            cur_block = self.basic_block(idim, odim, ksize)
            self.blocks.append(cur_block)
            if (i == 0) and self.do_downsample:
                idim = odim

    def forward(self, inp):
        """
        Args:
            inp: [B, idim, H, w]
        Returns:
            answer_label : [B, odim, H, w]
        """
        residual = inp
        out = None
        for i in range(self.nblocks):
            out = self.blocks[i](residual)
            if (i == 0) and self.do_downsample:
                residual = self.downsample(residual)
            out += residual
            out = F.relu(out) # w/o is sometimes better
            residual = out
        return out
    
    @staticmethod
    def basic_block(idim, odim, ksize=3):
        layers = []
        # 1st conv
        p = ksize // 2
        layers.append(nn.Conv1d(idim, odim, ksize, 1, p, bias=False))
        layers.append(nn.BatchNorm1d(odim))
        layers.append(nn.ReLU(inplace=True))
        # 2nd conv
        layers.append(nn.Conv1d(odim, odim, ksize, 1, p, bias=False))
        layers.append(nn.BatchNorm1d(odim))

        return nn.Sequential(*layers)


class NonLocalBlock(nn.Module):
    def __init__(self, idim, odim, num_heads, dropout, use_bias,
        use_local_mask, mask_kernel_size=15, mask_dilation=1
    ):
        super(NonLocalBlock, self).__init__()

        # dims
        self.idim = idim
        self.odim = odim
        self.nheads = num_heads

        # options
        self.use_local_mask = use_local_mask
        self.use_bias = use_bias
        if self.use_local_mask:
            self.ksize = mask_kernel_size
            self.dilation= mask_dilation
            self.local_mask = None

        # layers
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, m_feats, mask):
        """
        Inputs:
            m_feats: segment-level multimodal feature     [B,nseg,*]
            mask: mask                              [B,nseg]
        Outputs:
            updated_m: updated multimodal  feature  [B,nseg,*]
        """

        mask = mask.float()
        B, nseg = mask.size()

        # key, query, value
        m_k = self.v_lin(self.drop(m_feats)) # [B,num_seg,*]
        m_trans = self.c_lin(self.drop(m_feats))  # [B,nseg,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        # applying multi-head attention
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        r = None
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i] # [B, nseg, *]

            # compute relation matrix; [B,nseg,nseg]
            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            if self.use_local_mask:
                local_mask = mask.new_tensor(self._get_mask(nseg, self.ksize, self.dilation)) # [nseg,nseg]
                m2m = m2m.masked_fill(local_mask.unsqueeze(0).eq(0), -1e9) # [B,nseg,nseg]
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9) # [B,nseg,nseg]
            m2m_w = F.softmax(m2m, dim=2) # [B,nseg,nseg]
            w_list.append(m2m_w)

            # compute relation vector for each segment
            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)

        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)

    def _get_mask(self, N, ksize, d):
        if self.local_mask is not None: return self.local_mask
        self.local_mask = np.eye(N)
        K = ksize // 2
        for i in range(1, K+1):
            self.local_mask += np.eye(N, k=d+(i-1)*d)
            self.local_mask += np.eye(N, k=-(d+(i-1)*d))
        return self.local_mask # [N,N]


class AttentivePooling(nn.Module):
    def __init__(self, num_layer, feat_dim, hidden_dim, use_embedding, embedding_dim):
        super(AttentivePooling, self).__init__()

        self.att_n = num_layer
        self.feat_dim = feat_dim
        self.att_hid_dim = hidden_dim
        self.use_embedding = use_embedding

        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, self.att_n, bias=False)
        if self.use_embedding:
            self.fc = nn.Linear(self.feat_dim, embedding_dim)

    def forward(self, feats, f_masks=None):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            feats: features where attention weights are computed; [B, A, D]
            f_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert f_masks is None or len(f_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # embedding feature vectors
        attn_f = self.feat2att(feats)   # [B,A,hdim]

        # compute attention weights
        dot = torch.tanh(attn_f)        # [B,A,hdim]
        alpha = self.to_alpha(dot)      # [B,A,att_n]
        if f_masks is not None:
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  F.softmax(alpha.transpose(1,2), dim=2) # [B,att_n,A]

        att_feats = attw @ feats # [B,att_n,D]
        if self.att_n == 1:
            att_feats = att_feats.squeeze(1)
            attw = attw.squeeze(1)
        if self.use_embedding: att_feats = self.fc(att_feats)

        return att_feats, attw


class Attention(nn.Module):
    def __init__(self, kdim, cdim, att_hdim, drop_p):
        super(Attention, self).__init__()

        # layers
        self.key2att = nn.Linear(kdim, att_hdim)
        self.feat2att = nn.Linear(cdim, att_hdim)
        self.to_alpha = nn.Linear(att_hdim, 1)
        self.drop = nn.Dropout(drop_p)

    def forward(self, key, feats, feat_masks=None, return_weight=True):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2, "{} != 2".format(len(key.size()))
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # compute attention weights
        logits = self.compute_att_logits(key, feats, feat_masks) # [B,A]
        weight = self.drop(F.softmax(logits, dim=1))             # [B,A]

        # compute weighted sum: bmm working on (B,1,A) * (B,A,D) -> (B,1,D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        if return_weight:
            return att_feats, weight
        return att_feats

    def compute_att_logits(self, key, feats, feat_masks=None):
        """ Compute attention weights
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = self.apply_on_sequence(self.feat2att, feats)   # B * A * att_hdim
        att_k = self.key2att(key)                                   # B * att_hdim
        att_k = att_k.unsqueeze(1).expand_as(att_f)                 # B * A * att_hdim

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                             # B * A * att_hdim
        alpha = self.apply_on_sequence(self.to_alpha, dot)     # B * A * 1
        alpha = alpha.view(-1, A)                                   # B * A
        if feat_masks is not None:
            alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9)

        return alpha

    """ Computation helpers """
    @staticmethod
    def apply_on_sequence(layer, inp):
        " For nn.Linear, this fn is DEPRECATED "
        def to_contiguous(tensor):
            if tensor.is_contiguous():
                return tensor
            else:
                return tensor.contiguous()
        inp = to_contiguous(inp)
        inp_size = list(inp.size())
        output = layer(inp.view(-1, inp_size[-1]))
        output = output.view(*inp_size[:-1], -1)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    @staticmethod
    def softmax(x, dim, onnx_trace=False):
        if onnx_trace:
            return F.softmax(x.float(), dim=dim)
        else:
            return F.softmax(x, dim=dim, dtype=torch.float32)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # q:    bsz * num_heads, tgt_len, head_dim
        # k, v: bsz * num_heads, src_len, head_dim
        # key_padding_mask: bsz, src_len

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # attn_weights: bsz * num_heads, tgt_len, src_len
        # attn_mask:    tgt_len, src_len
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = self.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.gru.flatten_parameters()

    def forward(self, x, seq_len, max_num_frames):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            # print(sorted_idx)
            sorted_x = x.index_select(1, sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(), batch_first=self.batch_first)

        out, state = self.gru(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames:
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        # state = state.transpose(0, 1).contiguous().view(out.size(0), -1)
        return out


class GraphConvolution(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.num_relations = 40 ## 40 in CMIN original code, 44 in repro
        self.fc_dir_weight = self.clones(nn.Linear(d_model, d_model, bias=False), 3)
        self.fc_dir_bias = [nn.Parameter(torch.zeros(d_model))
                            for _ in range(self.num_relations * 2 - 1)]
        self.fc_dir_bias1 = nn.ParameterList(self.fc_dir_bias[-1:])
        self.fc_dir_bias2 = nn.ParameterList(self.fc_dir_bias[:self.num_relations - 1])
        self.fc_dir_bias3 = nn.ParameterList(self.fc_dir_bias[self.num_relations - 1:-1])

        self.fc_gate_weight = self.clones(nn.Linear(d_model, d_model, bias=False), 3)
        self.fc_gate_bias = [nn.Parameter(torch.zeros(d_model))
                             for _ in range(self.num_relations * 2 - 1)]
        self.fc_gate_bias1 = nn.ParameterList(self.fc_gate_bias[-1:])
        self.fc_gate_bias2 = nn.ParameterList(self.fc_gate_bias[:self.num_relations - 1])
        self.fc_gate_bias3 = nn.ParameterList(self.fc_gate_bias[self.num_relations - 1:-1])
    
    @ staticmethod
    def clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def _compute_one_direction(self, x, fc, biases, adj_mat, relations, fc_gate, biases_gate):
        x = fc(x)
        g = fc_gate(x)
        out = None
        for r, bias, bias_gate in zip(relations, biases, biases_gate):
            mask = (adj_mat == r).float()
            g1 = torch.sigmoid(g + bias_gate)
            res = torch.matmul(mask, (x + bias) * g1)
            if out is None:
                out = res
            else:
                out += res
        return out

    def forward(self, node, node_mask, adj_mat):
        out = self._compute_one_direction(node, self.fc_dir_weight[1], self.fc_dir_bias2,
                                          adj_mat, range(2, self.num_relations + 1),
                                          self.fc_gate_weight[1], self.fc_gate_bias2)
        adj_mat = adj_mat.transpose(-1, -2)
        out += self._compute_one_direction(node, self.fc_dir_weight[2], self.fc_dir_bias3,
                                           adj_mat, range(2, self.num_relations + 1),
                                           self.fc_gate_weight[2], self.fc_gate_bias3)
        # adj_mat = torch.eye(adj_mat.size(1)).type_as(adj_mat)
        out += self._compute_one_direction(node, self.fc_dir_weight[0], self.fc_dir_bias1,
                                           adj_mat, [1],
                                           self.fc_gate_weight[0], self.fc_gate_bias1)
        return F.relu(out)


class TanhAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0, direction=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)
        self.direction = direction

    def forward(self, x, memory, memory_mask=None):
        item1 = self.ws1(x)  # [nb, len1, d]
        item2 = self.ws2(memory)  # [nb, len2, d]
        # print(item1.shape, item2.shape)
        item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
        S = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S = S.masked_fill(memory_mask == 0, -1e30)
            # for forward, backward, S: [nb, len, len]
            if self.direction == 'forward':
                length = S.size(1)
                forward_mask = torch.ones(length, length)
                for i in range(1, length):
                    forward_mask[i, 0:i] = 0
                S = S.masked_fill(forward_mask.cuda().unsqueeze(0) == 0, -1e30)
            elif self.direction == 'backward':
                length = S.size(1)
                backward_mask = torch.zeros(length, length)
                for i in range(0, length):
                    backward_mask[i, 0:i + 1] = 1
                S = S.masked_fill(backward_mask.cuda().unsqueeze(0) == 0, -1e30)
        S = self.dropout(F.softmax(S, -1))
        return torch.matmul(S, memory)  # [nb, len1, d]


class CrossGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model, d_model, bias=False)
        self.fc_gate2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.fc_gate1(x1))
        x2_ = g1 * x2
        g2 = torch.sigmoid(self.fc_gate2(x2))
        x1_ = g2 * x1
        return x1_, x2_


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)].cuda(), 
                         requires_grad=False)
        #x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t.cuda())))



class CoAttention(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        self.q_proj = nn.Linear(512, 512, bias = False)
        #self.gate1 = nn.Conv1d(512, 1, kernel_size=1, bias = False)
        #self.gate_s1 = nn.Sigmoid()
        #self.gate2 = nn.Conv1d(512, 1, kernel_size=1, bias = False)
        #self.gate_s2 = nn.Sigmoid()

    def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
        Q = self.q_proj(x1)
        D = x2 #* (node_mask.unsqueeze(-1))

        D_t = torch.transpose(D, 1, 2).contiguous() #128, 512, 20
        L = torch.bmm(Q, D_t) #128, 200, 20

        Q_t = torch.transpose(Q, 1, 2).contiguous() # 128, 512, 200
        A_D = F.softmax(L, dim=2)
        C_D = torch.bmm(Q_t, A_D) # 128, 512, 20

        A_Q_ = F.softmax(L, dim = 1)
        A_Q = torch.transpose(A_Q_, 1, 2).contiguous() #128, 20, 200
        C_Q = torch.bmm(D_t, A_Q) # 128, 512, 200

        #C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        #C_D = C_D * self.gate_s2(self.gate2(C_D))

        C_Q = torch.transpose(C_Q, 1, 2).contiguous()
        C_D = torch.transpose(C_D, 1, 2).contiguous()

        # C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        # C_D = C_D * self.gate_s2(self.gate2(C_D))

        return C_Q, C_D, A_D, A_Q

class CoAttention_intra(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        #self.rte = RelTemporalEncoding(n_hid=d_model2,max_len=d_model1)
    def forward(self, x1, x2, node_mask): #128, 200, 512 128, 20, 512
        '''
        b, t, f = x1.shape[:]
        for tt in range(t):
            curr = x1[:,tt,:].unsqueeze(1).contiguous()
            time = torch.arange(0, t).unsqueeze(0).contiguous()
            time -= tt
            time = torch.abs(time)
            res = self.rte(x1,time)
            L = torch.bmm(curr, torch.transpose(res, 1, 2).contiguous())
            A = F.softmax(L, dim=2)
            x2[:,tt,:] = torch.bmm(A,res).squeeze(1).contiguous()
        return x2, x2, A, A
        '''
        Q = x1#PositionalEncoding(x1.shape[-1],0,x1.shape[-2])(x1)#self.q_proj(x1)
        D = x2#PositionalEncoding(x1.shape[-1],0,x1.shape[-2])(x2)#x2 #* (node_mask.unsqueeze(-1))

        D_t = torch.transpose(D, 1, 2).contiguous() #128, 512, 20
        #L = torch.bmm(Q, D_t)
        L = torch.bmm(PositionalEncoding(x1.shape[-1],0.2,x1.shape[-2])(Q), torch.transpose(PositionalEncoding(x1.shape[-1],0.2,x1.shape[-2])(D), 1, 2).contiguous()) #128, 200, 20

        Q_t = torch.transpose(Q, 1, 2).contiguous() # 128, 512, 200
        A_D = F.softmax(L, dim=2)
        C_D = torch.bmm(Q_t, A_D) # 128, 512, 20

        A_Q_ = F.softmax(L, dim = 1)
        A_Q = torch.transpose(A_Q_, 1, 2).contiguous() #128, 20, 200
        C_Q = torch.bmm(D_t, A_Q) # 128, 512, 200

        #C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        #C_D = C_D * self.gate_s2(self.gate2(C_D))

        C_Q = torch.transpose(C_Q, 1, 2).contiguous()
        C_D = torch.transpose(C_D, 1, 2).contiguous()

        # C_Q = C_Q * self.gate_s1(self.gate1(C_Q))
        # C_D = C_D * self.gate_s2(self.gate2(C_D))
       
        return C_Q, C_D, A_D, A_Q


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=1):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.out_gate = nn.Linear(input_size + hidden_size, hidden_size)

        torch.nn.init.orthogonal_(self.reset_gate.weight)
        torch.nn.init.orthogonal_(self.update_gate.weight)
        torch.nn.init.orthogonal_(self.out_gate.weight)
        torch.nn.init.constant_(self.reset_gate.bias, 0.)
        torch.nn.init.constant_(self.update_gate.bias, 0.)
        torch.nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, len]
        stacked_inputs = torch.cat([input_, prev_state], dim=2)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=2)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state