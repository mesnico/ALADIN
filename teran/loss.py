import torch
from torch import nn as nn
from torch.nn import functional as F
from .utils import l2norm
from math import sqrt


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class AlignmentContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, aggregation='sum-max-sentences'):
        super(AlignmentContrastiveLoss, self).__init__(margin, measure, max_violation)
        self.aggregation = aggregation

    def forward(self, im_set, s_seq, im_len, s_len, return_loss=True, return_similarity_mat=False):
        im_set = F.normalize(im_set, p=2, dim=2)
        s_seq = F.normalize(s_seq, p=2, dim=2)

        # im_set = im_set.permute(1, 0, 2)    # B x S_im x dim
        # s_seq = s_seq.permute(1, 0, 2)     # B x S_s x dim

        # do not consider cls and eos tokens
        im_set = im_set[:, 1:, :]
        s_seq = s_seq[:, 1:-2, :]
        im_len = [l - 1 for l in im_len]
        s_len = [l - 3 for l in s_len]

        im_set_batch = im_set.size(0)
        im_set_len = im_set.size(1)
        s_seq_batch = s_seq.size(0)
        s_seq_len = s_seq.size(1)

        im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
        s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
        # alignments = F.relu(alignments)

        # compute mask for the alignments tensor
        im_len_mask = torch.zeros(im_set_batch, im_set_len).bool()
        im_len_mask = im_len_mask.to(im_set.device)
        for im, l in zip(im_len_mask, im_len):
            im[l:] = True
        im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)

        s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool()
        s_len_mask = s_len_mask.to(im_set.device)
        for sm, l in zip(s_len_mask, s_len):
            sm[l:] = True
        s_len_mask = s_len_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1)

        alignment_mask = im_len_mask | s_len_mask
        alignments.masked_fill_(alignment_mask, value=0)
        # alignments = F.relu(alignments)
        # alignments = F.normalize(alignments,p=2, dim=2)

        if self.aggregation == 'sum':
            aggr_similarity = alignments.sum(dim=(2,3))
        elif self.aggregation == 'mean':
            aggr_similarity = alignments.mean(dim=(2,3))
        elif self.aggregation == 'MrSw':
            aggr_similarity = alignments.max(2)[0].sum(2)
        elif self.aggregation == 'MrAVGw':
            aggr_similarity = alignments.max(2)[0].sum(2)
            expanded_len = torch.FloatTensor(s_len).to(alignments.device).unsqueeze(0).expand(len(im_len), -1)
            aggr_similarity /= expanded_len
        elif self.aggregation == 'symm':
            im = alignments.max(2)[0].sum(2)
            s = alignments.max(3)[0].sum(2)
            aggr_similarity = im + s
        elif self.aggregation == 'MwSr':
            aggr_similarity = alignments.max(3)[0].sum(2)
        elif self.aggregation == 'scan-sentences':
            norm_alignments = F.relu(alignments)
            norm_alignments = F.normalize(norm_alignments,p=2, dim=2)
            weights = norm_alignments.masked_fill(alignment_mask, value=float('-inf'))
            weights = torch.softmax(weights, dim=3)

            weights = weights.unsqueeze(3)  # B x B x im x 1 x s
            s_seq_ext = s_seq.unsqueeze(2).expand(-1, -1, im_set_len, -1, -1)
            att_vector = torch.matmul(weights, s_seq_ext)  # B x B x im x 1 x dim
            att_vector = att_vector.squeeze(3)
            new_alignments = F.cosine_similarity(im_set, att_vector, dim=3)  # B x B x im
            new_alignments.masked_fill_(im_len_mask[:, :, :, 0], value=0)

            aggr_similarity = new_alignments.sum(2)

        if return_loss:
            loss = self.compute_contrastive_loss(aggr_similarity)

        if return_loss and return_similarity_mat:
            return loss, aggr_similarity
        elif return_loss:
            return loss
        elif return_similarity_mat:
            return aggr_similarity


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s, return_similarity_mat=False):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        loss = self.compute_contrastive_loss(scores)
        if return_similarity_mat:
            return loss, scores
        else:
            return loss



class CrossEntropyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, im, s):
        logits = im.mm(s.t()) * torch.exp(self.temperature)
        labels = torch.arange(im.shape[0]).to(im.device)
        loss_i = F.cross_entropy(logits.t(), labels)
        loss_t = F.cross_entropy(logits, labels)
        loss = (loss_i + loss_t) / 2
        return loss

class SemanticContrastiveLoss(nn.Module):
    def __init__(self, margin=0, threshold=0.4, measure=False, max_violation=False):
        super().__init__()
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim
        self.margin = margin
        self.max_violation = max_violation
        self.threshold = threshold

    def compute_semantic_contrastive_loss(self, scores, relevances):
        # diagonal = scores.diag().view(scores.size(0), 1)
        # d1 = diagonal.expand_as(scores)
        # d2 = diagonal.t().expand_as(scores)

        matching_mask = relevances > self.threshold  # B x B of boolean flags

        rows_mask = torch.zeros_like(matching_mask).bool()
        for i in range(matching_mask.shape[0]):
            non_zeros_idxs = torch.nonzero(matching_mask[i, :])
            r = torch.randint(non_zeros_idxs.shape[0], (1,))
            random_col_idx = non_zeros_idxs[r]
            rows_mask[i, random_col_idx] = True
        d1 = scores[rows_mask].view(scores.size(0), 1).expand_as(scores)

        cols_mask = torch.zeros_like(matching_mask).bool()
        for i in range(matching_mask.shape[1]):
            non_zeros_idxs = torch.nonzero(matching_mask[:, i])
            r = torch.randint(non_zeros_idxs.shape[0], (1,))
            random_row_idx = non_zeros_idxs[r]
            cols_mask[random_row_idx, i] = True
        d2 = scores[cols_mask].view(scores.size(0), 1).t().expand_as(scores)

        # TODO: zero out the non-chosen ones!

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def forward(self, im, s, relevances, return_similarity_mat=False):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        loss = self.compute_semantic_contrastive_loss(scores, relevances)
        if return_similarity_mat:
            return loss, scores
        else:
            return loss


class AttentionDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im_set, s_seq, im_len, s_len, teacher_attentions):
        # alignment_mat: im x sen x r x w
        # teacher_att: im x sen x w(69) x r(70)

        # do not consider cls and eos tokens
        im_set = im_set[:, 1:, :]
        s_seq = s_seq[:, 1:, :]
        im_len = [l - 1 for l in im_len]
        s_len = [l - 1 for l in s_len]
        k = im_set.shape[2]

        im_set_batch = im_set.size(0)
        im_set_len = im_set.size(1)
        s_seq_batch = s_seq.size(0)
        s_seq_len = s_seq.size(1)

        im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
        s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1)  # B x B x S_s x dim
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
        alignments = alignments / sqrt(k)   # scaled dot product

        # compute mask for the alignments tensor
        im_len_mask = torch.zeros(im_set_batch, im_set_len).bool()
        im_len_mask = im_len_mask.to(im_set.device)
        for im, l in zip(im_len_mask, im_len):
            im[l:] = True
        im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)

        s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool()
        s_len_mask = s_len_mask.to(im_set.device)
        for sm, l in zip(s_len_mask, s_len):
            sm[l:] = True
        s_len_mask = s_len_mask.unsqueeze(0).expand(im_set_batch, -1, -1)

        alignment_mask = im_len_mask # | s_len_mask
        #alignments.masked_fill_(alignment_mask, value=0)

        alignments = alignments.permute(0, 1, 3, 2) # im x sen x w x r
        alignment_mask = alignment_mask.permute(0, 1, 3, 2) # im x sen x w x r

        # transform both alignments and gt to probabilities
        alignments.masked_fill_(alignment_mask, value=float('-inf'))
        alignments = torch.log_softmax(alignments, dim=-1)
        teacher_attentions = teacher_attentions[:, :, :s_seq_len, :im_set_len]
        # teacher_attentions.masked_fill_(alignment_mask, value=float('-inf'))
        # teacher_attentions = torch.softmax(teacher_attentions, dim=-1)

        # renormalize the gt distribution
        teacher_attentions = F.normalize(teacher_attentions, p=1, dim=3)

        # mask over the words dimension
        teacher_attentions = teacher_attentions[~s_len_mask]
        alignments = alignments[~s_len_mask]

        # alignments = alignments.flatten(start_dim=0, end_dim=2)
        # teacher_attentions = teacher_attentions.flatten(start_dim=0, end_dim=2)
        loss = F.kl_div(alignments, teacher_attentions, reduction='batchmean')
        return loss



class PermInvMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # @staticmethod
    # def batched_cosine_sim(im, s):
    #     """Cosine similarity between all the image and sentence pairs
    #     """
    #     im = F.normalize(im, p=2, dim=2)
    #     s = F.normalize(s, p=2, dim=2)
    #     return im.mm(s.permute(0, 2, 1))

    def forward(self, im, s):
        dist_matrix = torch.cdist(im, s, p=2)
        row_sum = F.softmin(dist_matrix, dim=2).max(dim=2)[0].sum(dim=1)
        col_sum = F.softmin(dist_matrix, dim=1).max(dim=1)[0].sum(dim=1)
        loss = 2*torch.Tensor([dist_matrix.shape[1]]).to(im.device) - row_sum - col_sum
        loss = loss.mean()
        return loss


class DistillationLoss(nn.Module):
    def __init__(self, mode='mse', margin=0.2, threshold=0.1, stride=3):
        super().__init__()
        self.mode = mode
        self.margin = margin
        self.threshold = threshold
        self.stride = stride
        if mode == 'mse':
            self.wb = nn.Parameter(torch.FloatTensor([0.5, 0.5]), requires_grad=True)

    def forward(self, teacher_scores, student_scores):
        teacher_scores = teacher_scores.detach()    # do not backprop through the teacher branch
        if self.mode == 'mse':
            student_scores = student_scores * self.wb[0] + self.wb[1] # (student_scores + 1) / 2
            loss = F.mse_loss(student_scores.view(-1).unsqueeze(1), teacher_scores.view(-1).unsqueeze(1))
        elif self.mode == 'ordinal':
            # do not propagate the gradients directly through the teacher pipeline
            # teacher_scores = teacher_scores.detach()
            # in this case, the order established by the teacher should respect the one in the student, both in rows and in columns (image and text search)

            # image to caption search (by row)
            teacher_scores_row, teacher_rankings_row = torch.sort(teacher_scores, dim=1) # order every row
            student_ordered_row = torch.gather(student_scores, dim=1, index=teacher_rankings_row)
            student_differences_row = student_ordered_row[:, :-self.stride] - student_ordered_row[:, self.stride:]  # the differences should be all negatives
            # gradients only for appreciable differences
            valid = teacher_scores_row >= self.threshold
            student_differences_row = student_differences_row[valid[:, self.stride:]]
            # compute per-row loss
            ordinal_loss_rows = F.relu(self.margin + student_differences_row).mean()

            # caption to image search (by column)
            teacher_scores_col, teacher_rankings_col = torch.sort(teacher_scores, dim=0) # order every column
            student_ordered_col = torch.gather(student_scores, dim=0, index=teacher_rankings_col)
            student_differences_col = student_ordered_col[:-self.stride, :] - student_ordered_col[self.stride:, :]
            # gradients only for appreciable differences
            valid = teacher_scores_col >= self.threshold
            student_differences_col = student_differences_col[valid[self.stride:, :]]
            # compute per-col loss
            ordinal_loss_cols = F.relu(self.margin + student_differences_col).mean()

            loss = ordinal_loss_rows + ordinal_loss_cols

        elif self.mode == 'contrastive':
            mask = torch.eye(teacher_scores.size(0)) > .5
            if torch.cuda.is_available():
                mask = mask.cuda()
            teacher_scores_nodiag = teacher_scores.detach().masked_fill_(mask, 0)

            diagonal = student_scores.diag().view(student_scores.size(0), 1)
            d1 = diagonal.expand_as(student_scores)
            d2 = diagonal.t().expand_as(student_scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + student_scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.margin + student_scores - d2).clamp(min=0)

            # keep the maximum violating negative for each query, taking the hard negative from the teacher scores
            negative_idx_s = teacher_scores_nodiag.max(1)[1]
            cost_s = cost_s.index_select(dim=1, index=negative_idx_s)

            negative_idxs_im = teacher_scores_nodiag.max(0)[1]
            cost_im = cost_im.index_select(dim=0, index=negative_idxs_im)

            loss = cost_s.sum() + cost_im.sum()

        elif self.mode == 'listnet':
            eps = 1e-10
            temperature = 6.0

            # sentence retrieval
            preds_smax = F.softmax(student_scores * temperature, dim=1)
            true_smax = F.softmax(teacher_scores, dim=1)
            preds_smax = preds_smax + eps
            preds_log = torch.log(preds_smax)
            s_cost = torch.mean(-torch.sum(true_smax * preds_log, dim=1))

            # image retrieval
            preds_smax = F.softmax(student_scores * temperature, dim=0)
            true_smax = F.softmax(teacher_scores, dim=0)
            preds_smax = preds_smax + eps
            preds_log = torch.log(preds_smax)
            im_cost = torch.mean(-torch.sum(true_smax * preds_log, dim=0))

            loss = im_cost + s_cost

        return loss
