from __future__ import print_function

import numpy
# from speaksee.data import COCO, RawField
from torch.utils.data import DataLoader

# from teran import data
import itertools
# from teran.data import get_test_loader
import time
import numpy as np
import torch
import tqdm
from collections import OrderedDict

# from recall_auxiliary import recall_1k_5fold_test
# from utils import dot_sim, get_model, ImageDetectionsField
from alad.evaluate_utils.dcg import DCG
from alad.loss import order_sim, AlignmentContrastiveLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    img_lengths = []
    cap_lengths = []

    # compute maximum lenghts in the whole dataset
    max_cap_len = 71
    max_img_len = 71
    # for _, _, img_length, cap_length, _, _ in data_loader:
    #     max_cap_len = max(max_cap_len, max(cap_length))
    #     max_img_len = max(max_img_len, max(img_length))

    ids_pointer = 0
    for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
        example_imgs, example_txts = batch_data

        # make sure val logger is used
        model.logger = val_logger
        bs = example_imgs[0].shape[0]
        ids = list(range(ids_pointer, ids_pointer + bs))
        ids_pointer += bs

        # compute the embeddings
        with torch.no_grad():
            img_cross_attention, cap_cross_attention, img_emb, cap_emb, img_length, cap_length, _ = model.forward_emb(example_imgs, example_txts)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = torch.zeros((len(data_loader.dataset), max_img_len, img_emb.size(2)))
                cap_embs = torch.zeros((len(data_loader.dataset), max_cap_len, cap_emb.size(2)))

            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[ids, :img_emb.size(0), :] = img_emb.cpu().permute(1, 0, 2)
            cap_embs[ids, :cap_emb.size(0), :] = cap_emb.cpu().permute(1, 0, 2)
            # insert global embs as T-CLS and I-CLS tokens
            img_embs[ids, 0, :] = img_cross_attention.cpu()
            cap_embs[ids, 0, :] = cap_cross_attention.cpu()
            img_lengths.extend(img_length)
            cap_lengths.extend(cap_length)

            # measure accuracy and record loss
            # model.forward_loss(None, None, img_emb, cap_emb, img_length, cap_length)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del batch_data

    # p = np.random.permutation(len(data_loader.dataset) // 5) * 5
    # p = np.transpose(np.tile(p, (5, 1)))
    # p = p + np.array([0, 1, 2, 3, 4])
    # p = p.flatten()
    # img_embs = img_embs[p]
    # cap_embs = cap_embs[p]

    return img_embs, cap_embs, img_lengths, cap_lengths


def i2t(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None, cap_batches=1):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    rougel_ndcgs = numpy.zeros(npts)
    spice_ndcgs = numpy.zeros(npts)
    # captions = captions.cuda()
    captions_per_batch = captions.shape[0] // cap_batches

    for index in tqdm.trange(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1], images.shape[2])
        im = im.cuda() if sim_function is not None else im
        im_len = [img_lenghts[5 * index]]

        d = None

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            if sim_function is None:
                d = torch.mm(im[:, 0, :], captions[:, 0, :].t())
                d = d.cpu().numpy().flatten()
            else:
                for i in range(cap_batches):
                    captions_now = captions[i*captions_per_batch:(i+1)*captions_per_batch]
                    cap_lenghts_now = cap_lenghts[i*captions_per_batch:(i+1)*captions_per_batch]
                    captions_now = captions_now.cuda()

                    d_align = sim_function(im, captions_now, im_len, cap_lenghts_now)
                    d_align = d_align.cpu().numpy().flatten()
                    # d_matching = torch.mm(im[:, 0, :], captions[:, 0, :].t())
                    # d_matching = d_matching.cpu().numpy().flatten()
                    if d is None:
                        d = d_align # + d_matching
                    else:
                        d = numpy.concatenate([d, d_align], axis=0)

        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if ndcg_scorer is not None:
            rougel_ndcgs[index], spice_ndcgs[index] = ndcg_scorer.compute_ndcg(npts, index, inds.astype(int),
                                                                               fold_index=fold_index,
                                                                               retrieval='sentence').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = 0
    mean_spice_ndcg = 0
    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)


def t2i(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None, im_batches=1):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = torch.stack([images[i] for i in range(0, len(images), 5)], dim=0)
    # ims = ims.cuda()
    ims_len = [img_lenghts[i] for i in range(0, len(images), 5)]

    ranks = numpy.zeros(5 * npts)
    top50 = numpy.zeros((5 * npts, 50))
    rougel_ndcgs = numpy.zeros(5 * npts)
    spice_ndcgs = numpy.zeros(5 * npts)

    images_per_batch = ims.shape[0] // im_batches

    for index in tqdm.trange(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        queries = queries.cuda() if sim_function is not None else queries
        queries_len = cap_lenghts[5 * index:5 * index + 5]

        d = None

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            if sim_function is None:
                d = torch.mm(queries[:, 0, :], ims[:, 0, :].t())
                d = d.cpu().numpy()
            else:
                for i in range(im_batches):
                    ims_now = ims[i * images_per_batch:(i+1) * images_per_batch]
                    ims_len_now = ims_len[i * images_per_batch:(i+1) * images_per_batch]
                    ims_now = ims_now.cuda()

                    # d = numpy.dot(queries, ims.T)
                    d_align = sim_function(ims_now, queries, ims_len_now, queries_len).t()
                    d_align = d_align.cpu().numpy()
                    # d_matching = torch.mm(queries[:, 0, :], ims[:, 0, :].t())
                    # d_matching = d_matching.cpu().numpy()
                    if d is None:
                        d = d_align # + d_matching
                    else:
                        d = numpy.concatenate([d, d_align], axis=1)

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][
                0]  # in che posizione e' l'immagine (index) che ha questa caption (5*index + i)
            top50[5 * index + i] = inds[i][0:50]
            # calculate ndcg
            if ndcg_scorer is not None:
                rougel_ndcgs[5 * index + i], spice_ndcgs[5 * index + i] = \
                    ndcg_scorer.compute_ndcg(npts, 5 * index + i, inds[i].astype(int),
                                             fold_index=fold_index, retrieval='image').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = 0
    mean_spice_ndcg = 0

    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top50)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)

