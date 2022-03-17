import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScoreDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(ScoreDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, 1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, 1)
        return out


class MultiHeadAttentionAggregation(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, N_aggregated_vectors, kv_score_div=4):
        super(MultiHeadAttentionAggregation, self).__init__()

        self.aggregating_layers = nn.ModuleList([ScoreDotProductAttention(d_model=d_model, d_k=d_k // kv_score_div,
                                                                          d_v=d_v // kv_score_div, h=h)
                                                 for _ in range(N_aggregated_vectors)])

    def forward(self, queries, cross_keys, cross_values, attention_mask=None, cross_attention_mask=None):
        #if attention_mask is not None:
            # attention_mask = (1-(attention_mask.byte())).squeeze(1).permute(0,2,1) # prepare to 0/1 boolean mask
        #    attention_mask = (1 - (attention_mask.byte()))
        attention_mask = attention_mask.unsqueeze(2)
        attention_mask_float = (1 - (attention_mask.byte())).float()

        new_out = list()
        for layer in self.aggregating_layers:
            if attention_mask is not None:
                att = layer(queries, cross_keys, cross_values, cross_attention_mask)
                att = att.masked_fill(attention_mask, float("-inf"))
                att = torch.softmax(att, dim=1)
                aggregated_vec_i =  queries * att
                new_out.append((torch.sum(aggregated_vec_i * attention_mask_float, dim=1) / torch.sum(attention_mask_float, dim=1)).unsqueeze(1))  # media maskerata
            else:
                aggregated_vec_i = queries * torch.softmax(layer(queries, cross_keys, cross_values, cross_attention_mask), dim=1)
                new_out.append(aggregated_vec_i.sum(1, keepdim=True))  # media non maskerata

        new_out = torch.cat(new_out, 1)

        return new_out


class SelfAggregation(nn.Module):
    def __init__(self, d_model, feedforward_dim):
        super().__init__()
        self.txt_multi_head_att_aggregation = MultiHeadAttentionAggregation(d_model=d_model, d_k=feedforward_dim, d_v=feedforward_dim, h=4, N_aggregated_vectors=1, kv_score_div=1)
        self.img_multi_head_att_aggregation = MultiHeadAttentionAggregation(d_model=d_model, d_k=feedforward_dim, d_v=feedforward_dim, h=4, N_aggregated_vectors=1, kv_score_div=1)

    def forward(self, img_emb_set, cap_emb_seq, img_lengths, cap_lengths):
        bs = img_emb_set.shape[0]

        # do not consider cls tokens (reserved)
        img_emb_set = img_emb_set[:, 1:, :]
        cap_emb_seq = cap_emb_seq[:, 1:, :]
        img_lengths = [l - 1 for l in img_lengths]
        cap_lengths = [l - 1 for l in cap_lengths]

        txt_mask = torch.zeros(bs, max(cap_lengths)).bool()
        txt_mask = txt_mask.to(img_emb_set.device)
        for m, c_len in zip(txt_mask, cap_lengths):
            m[c_len:] = True

        img_mask = torch.zeros(bs, max(img_lengths)).bool()
        img_mask = img_mask.to(img_emb_set.device)
        for m, i_len in zip(img_mask, img_lengths):
            m[i_len:] = True

        # compute cross-attention masks
        # im_set_len = img_emb_set.size(1)
        # s_seq_len = cap_emb_seq.size(1)
        # img_mask_ext = img_mask.unsqueeze(2).expand(-1, -1, s_seq_len)
        # txt_mask_ext = txt_mask.unsqueeze(1).expand(-1, im_set_len, -1)
        # img_cross_mask = img_mask_ext | txt_mask_ext
        # img_cross_mask = img_cross_mask.unsqueeze(1)
        # txt_cross_mask = img_cross_mask.permute(0, 1, 3, 2)

        # Aggregation with self, no cross contamination:
        txt_aggregated_feat = self.txt_multi_head_att_aggregation(cap_emb_seq, cap_emb_seq, cap_emb_seq, attention_mask=txt_mask, cross_attention_mask=txt_mask)
        img_aggregated_feat = self.img_multi_head_att_aggregation(img_emb_set, img_emb_set, img_emb_set, attention_mask=img_mask, cross_attention_mask=img_mask)
        #
        # Original with cross aggregation
        # txt_aggregated_feat = self.txt_multi_head_att_aggregation(cap_emb_seq, img_emb_set, img_emb_set, attention_mask=txt_mask, cross_attention_mask=img_mask)
        # img_aggregated_feat = self.img_multi_head_att_aggregation(img_emb_set, cap_emb_seq, cap_emb_seq, attention_mask=img_mask, cross_attention_mask=txt_mask)

        txt_aggregated_feat = txt_aggregated_feat.squeeze(1)
        img_aggregated_feat = img_aggregated_feat.squeeze(1)

        img_aggregated_feat = F.normalize(img_aggregated_feat, p=2, dim=1)
        txt_aggregated_feat = F.normalize(txt_aggregated_feat, p=2, dim=1)

        return img_aggregated_feat, txt_aggregated_feat