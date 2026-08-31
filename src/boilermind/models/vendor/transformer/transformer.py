import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
# from tst.encoder import Encoder
# from tst.decoder import Decoder
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding
from mytools import FFT_for_Period
from TCN import *
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, n_past, n_future, d_model, d_ff, num_heads, num_layers, dropout, top_k):
        super(transformer, self).__init__()
        self.k = top_k
        self.n_past = n_past
        self.n_future = n_future
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.freq = 't'
        self.pred_len = n_future
        self.factor = 1
        self.output_attention = 'true'
        self.x_mask = None
        self.predict_linear1 = nn.Linear(30, 1, bias=True) ############
        ##############################TCN###############################
        self.dropout = nn.Dropout(dropout)
        self.TCN_embedding = DataEmbedding(30, d_model, self.embed, self.freq, dropout)
        self.add_norm1 = AddNorm([n_past, 30], dropout)
        self.add_norm2 = AddNorm([n_past, d_model], dropout)
        ###################
        self.TCN = TemporalConvNet(30, [15, 30])
        #################################################################
        # Embedding
        self.enc_embedding = DataEmbedding(30, d_model, self.embed, self.freq, dropout)  ###########
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=dropout,output_attention=self.output_attention), d_model, num_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=self.activation
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.encoder = nn.ModuleList([Encoder(d_model,
        #                                               q,
        #                                               v,
        #                                               h,
        #                                               attention_size=num_heads,
        #                                               dropout=dropout,
        #                                               chunk_mode=None) for _ in range(num_layers)])
        # Decoder
        self.dec_embedding = DataEmbedding(1, d_model, self.embed, self.freq, dropout)     ############
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, self.factor, attention_dropout=dropout, output_attention=False), d_model, num_heads),
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=dropout, output_attention=False), d_model, num_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=self.activation,
                )
                for l in range(num_layers)
                ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 30, bias=True)############
        )
        # self.decoder = nn.ModuleList([Decoder(d_model,
        #                                               q,
        #                                               v,
        #                                               h,
        #                                               attention_size=num_heads,
        #                                               dropout=dropout,
        #                                               chunk_mode=None) for _ in range(num_layers)])


    def forecast1(self, x_enc, x_dec):
        # Embedding
        out_x = x_enc[:, -self.n_future, :]
        # period, _ = FFT_for_Period(x_enc, self.k)
        # period = torch.from_numpy(period).float()
        enc_out = self.enc_embedding(x_enc, self.x_mask)

        ##
        #
        #
        # TCN_needX = x_enc.permute(1, 2, 0)
        # TCN_out = TCN_needX
        # for p_num in range(len(period)):
        #     p_len = int(period[p_num])
        #     train_dataset = TensorDataset(TCN_needX)
        #     TCN_loader = DataLoader(train_dataset, batch_size=p_len, shuffle=True)
        #     TCN_o = []
        #     TCN_o = torch.tensor(TCN_o)
        #     for TCN_idx, TCN_ids in enumerate(TCN_loader):
        #         TCN_so = self.TCN(TCN_ids[0])
        #         TCN_o = torch.cat((TCN_o.cuda(), TCN_so.cuda()), dim=0)
        #     TCN_out = self.add_norm1(TCN_out.permute(2, 0, 1), TCN_o.permute(2, 0, 1))
        #     TCN_out = TCN_out.permute(1, 2, 0)
        #
        # TCN_out = self.TCN_embedding(TCN_out.permute(2, 0, 1), self.x_mask, period)
        # enc_out = self.add_norm2(TCN_out.cuda(), enc_out.cuda())

        # for layer in self.encoder:
        #     enc_out = layer(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(x_dec, self.x_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # for layer in self.decoder:
        #     dec_out = layer(dec_out, enc_out)
        dec_out = self.predict_linear1(dec_out.squeeze(1) + out_x)

        return F.gelu(dec_out)


    def forward(self, x_enc, x_dec):
        dec_out = self.forecast1(x_enc, x_dec)
        # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out.unsqueeze(1)
        # retrn dec_out
