import torch
from torch import nn
from torch.functional import F

from .BUModules import Attention, GraphConvolution, DynamicGRU
from ..Transformer import LayerNorm, AdaptiveSelfGatingAttention, PositionalEncoding


class LGIQueryEncoder(nn.Module):
    def __init__(self, 
        rnn_type, wordvec_dim, query_hidden_dim, 
        num_rnn_layers, dropout, bidirectional,
        num_semantic_entity, sqan_dropout
        ):
        super(LGIQueryEncoder, self).__init__()
        # Parameters
        self.bidirectional = bidirectional
        self.query_hidden_dim = query_hidden_dim
        self.nse = num_semantic_entity

        # RNN
        rnn_hidden_dim = query_hidden_dim // 2 if bidirectional else query_hidden_dim
        self.rnn = getattr(nn, rnn_type)(
            input_size=wordvec_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # SQAN
        if self.nse > 1:
            self.global_emb_fn = nn.ModuleList( # W_q^(n) in Eq. (4)
                    [nn.Linear(query_hidden_dim, query_hidden_dim) for _ in range(self.nse)]
            )
            self.guide_emb_fn = nn.Sequential(*[
                nn.Linear(2*query_hidden_dim, query_hidden_dim), # W_g in Eq. (4)
                nn.ReLU()
            ])
            self.att_fn = Attention(
                query_hidden_dim, query_hidden_dim, 
                query_hidden_dim//2, sqan_dropout
            )

    def forward(self, queries, wordlens, word_masks):
        """ encode query sequence
        Args:
            queries (tensor[B, maxL, w2v_dim])
            wordlens (tensor[B])
            word_masks (tensor[B, maxL])
        Returns:
            word_feats (tensor[B, maxL, query_hdim])
            sentence_feats (tensor[B, query_hdim])
            semantic_features (tensor[B, num_semantic_entity, query_hdim])
            semantic_attention_weights (tensor[B, num_semantic_entity, maxL])
        """
        self.rnn.flatten_parameters()
        B = queries.shape[0]
        H = self.query_hidden_dim
        # RNN Encoder
        if self.bidirectional:
            queries_packed = nn.utils.rnn.pack_padded_sequence(
                queries, wordlens, batch_first=True, enforce_sorted=False)
            word_feats = self.rnn(queries_packed)[0]
            word_feats, _ = nn.utils.rnn.pad_packed_sequence(
                word_feats, batch_first=True, total_length=queries.size(1))
            word_feats = word_feats.contiguous()
            fLSTM = word_feats[range(B), wordlens.long() - 1, :H//2] # [B, hdim/2]
            bLSTM = word_feats[:, 0, H//2:].view(B, H//2) # [B, hdim/2]
            sentence_feats = torch.cat([fLSTM, bLSTM], dim=1)
        else:
            word_feats = self.rnn(queries)[0]
            sentence_feats = word_feats[range(B), wordlens.long() - 1]
        
        # Sequential Query Attention Parser
        if self.nse > 1:
            prev_se = word_feats.new_zeros(B, H)
            se_feats, se_attw = [], []
            # compute semantic entity features sequentially
            for n in range(self.nse):
                # perform Eq. (4)
                q_n = self.global_emb_fn[n](sentence_feats) # [B,qdim] -> [B,qdim]
                g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1)) # [B,2*qdim] -> [B,qdim]
                # perform Eq. (5), (6), (7)
                att_f, att_w = self.att_fn(g_n, word_feats, word_masks)

                prev_se = att_f
                se_feats.append(att_f)
                se_attw.append(att_w)
            semantic_features = torch.stack(se_feats, dim=1)
            semantic_attention_weights = torch.stack(se_attw, dim=1)
        else:
            semantic_features = None
            semantic_attention_weights = None

        return {
            'word_features': word_feats, 
            'sentence_features': sentence_feats,
            'semantic_features': semantic_features,
            'semantic_attention_weights': semantic_attention_weights,
        }


class CMINQueryEncoder(nn.Module):
    def __init__(self, wordvec_dim, max_gcn_layers, model_hidden_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(wordvec_dim)
            for _ in range(max_gcn_layers)
        ])
        self.rnn = DynamicGRU(wordvec_dim, model_hidden_dim >> 1, bidirectional=True, batch_first=True)

    def forward(self, queries, wordmasks, querylen, adj_mats):
        """
            queries (tensor[B, maxL, w2v_dim])
            querylen (tensor[B])
            wordmasks (tensor[B, maxL])
            adj_mats (tensor[B, maxl, maxL])
        """
        queries = F.dropout(queries, self.dropout, self.training)

        for gcn in self.gcn_layers:
            res = queries
            queries = gcn(queries, wordmasks, adj_mats)
            queries = F.dropout(queries, self.dropout, self.training)
            queries = res + queries

        queries = self.rnn(queries, querylen, queries.shape[1])
        queries = F.dropout(queries, self.dropout, self.training)

        return queries


class FIANQueryEncoder(nn.Module):
    def __init__(self, wordvec_dim, model_hidden_dim):
        super().__init__()
        self.rnn = DynamicGRU(wordvec_dim, model_hidden_dim >> 1, bidirectional=True, batch_first=True)

    def forward(self, queries, querylen):
        """
            queries (tensor[B, maxL, w2v_dim])
            querylen (tensor[B])
            wordmasks (tensor[B, maxL])
        """
        queries = self.rnn(queries, querylen, queries.shape[1])

        return queries


class CSMGANQueryEncoder(nn.Module):
    def __init__(self, wordvec_dim, model_hidden_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.unigram_conv = nn.Conv1d(wordvec_dim, wordvec_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(wordvec_dim, wordvec_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(wordvec_dim, wordvec_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        # self.tanh = nn.Tanh()
        self.bilstm = nn.LSTM(input_size=wordvec_dim,
                              hidden_size=wordvec_dim // 2,
                              num_layers=2,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)
        self.concat = DynamicGRU(wordvec_dim*3, model_hidden_dim >> 1, bidirectional=True, batch_first=True)

    def forward(self, queries, querylen):
        """
            queries (tensor[B, maxL, w2v_dim])
            querylen (tensor[B])
        """
        words = queries.permute(0, 2, 1) #128, 300, 20
        unigrams = torch.unsqueeze(self.unigram_conv(words), 2) # B x 512 x L
        bigrams  = torch.unsqueeze(self.bigram_conv(words), 2)  # B x 512 x L
        trigrams = torch.unsqueeze(self.trigram_conv(words), 2) # B x 512 x L
        words = words.permute(0, 2, 1) #128, 20, 300

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1) #128, 20, 300

        self.bilstm.flatten_parameters()
        sentence, _ = self.bilstm(phrase)
        
        concate = torch.cat((words, phrase, sentence), 2)
        queries = self.concat(concate, querylen, queries.shape[1])
        queries = F.dropout(queries, self.dropout, self.training)

        return queries


def build_query_encoder(cfg, arch):
    wordvec_dim = cfg.INPUT.PRE_QUERY_SIZE
    if arch == 'LGI':
        rnn_type = cfg.MODEL.LGI.QUERYENCODER.RNN.TYPE
        query_hidden_dim = cfg.MODEL.LGI.QUERYENCODER.RNN.HIDDEN_SIZE
        num_rnn_layers = cfg.MODEL.LGI.QUERYENCODER.RNN.NUM_LAYERS
        dropout = cfg.MODEL.LGI.QUERYENCODER.RNN.DROPOUT
        bidirectional = cfg.MODEL.LGI.QUERYENCODER.RNN.BIDIRECTIONAL
        num_semantic_entity = cfg.MODEL.LGI.QUERYENCODER.NUM_SEMANTIC_ENTITY
        sqan_dropout = cfg.MODEL.LGI.QUERYENCODER.SQAN_DROPOUT
        return LGIQueryEncoder(
            rnn_type, wordvec_dim, query_hidden_dim, 
            num_rnn_layers, dropout, bidirectional,
            num_semantic_entity, sqan_dropout
        )
    elif arch == 'CMIN':
        model_hidden_dim = cfg.MODEL.CMIN.HIDDEN_DIM
        dropout = cfg.MODEL.CMIN.DROPOUT
        max_gcn_layers = cfg.MODEL.CMIN.QUERYENCODER.NUM_GCN_LAYERS
        return CMINQueryEncoder(
            wordvec_dim, max_gcn_layers, model_hidden_dim, dropout
        )
    elif arch == 'FIAN':
        model_hidden_dim = cfg.MODEL.FIAN.QUERYENCODER.HIDDEN_DIM
        return FIANQueryEncoder(
            wordvec_dim, model_hidden_dim
        )
    elif arch == 'CSMGAN':
        model_hidden_dim = cfg.MODEL.CSMGAN.HIDDEN_DIM
        dropout = cfg.MODEL.CSMGAN.DROPOUT
        return CSMGANQueryEncoder(
            wordvec_dim, model_hidden_dim, dropout
        )
    else:
        raise NotImplementedError