import numpy as np
import torch
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image: 5000 x 1024
# captions: 5000 x 1024
def recall(images, captions, model, mode='i2t', lenghts=None, return_ranks=False):
    b_s = images.size(0)
    im_flatten = images.view(b_s, -1)
    # indexes_b = 1 - torch.all(im_flatten[:-1] == im_flatten[1:], -1) # (b_s, )
    # indexes_b = ~torch.all(im_flatten[:-1] == im_flatten[1:], -1)  # (b_s, )
    # indexes_b = torch.cat([torch.ones(1, ).to(indexes_b.device).byte(), indexes_b.byte(), torch.ones(1, ).to(indexes_b.device).byte()])
    indexes_b = torch.zeros(im_flatten.shape[0] + 1).byte().to(im_flatten.device)
    indexes_b[0::5] = 1

    npts = torch.sum(indexes_b).item() - 1
    indexes = torch.nonzero(indexes_b).squeeze(-1) # (npts, )

    ranks = np.zeros(npts) if mode is 'i2t' else np.zeros(indexes_b.shape[0] - 1)
    top1 = np.zeros(npts) if mode is 'i2t' else np.zeros(indexes_b.shape[0] - 1)

    caps = captions
    if isinstance(images, list) or isinstance(images, tuple):
        ims = [i[indexes[:-1]] for i in images]
    else:
        ims = images[indexes[:-1]]

    if mode is 'i2t':
        d_arr = ims.mm(caps.t()).cpu().numpy() # model.similarity(ims, caps, lenghts).detach().cpu().numpy()  #todo detach is needed?
        # questo perchè il modello con prima func attention non ha il for sul batch -> devo fare similarity con tutte images
        # d_arr = model.similarity(images, caps, lenghts).cpu().detach().numpy()
        # d_arr = d_arr[indexes[:-1].cpu()]  # poi tolgo quelle ripetute (le stesse immagini)
        for img in trange(npts):
            d = d_arr[img].flatten()
            inds = np.argsort(d)[::-1]  # indexes that sort similarities from largest to smallest
            # find the best rank position among the 5 captions ground-truth of the image
            rank = 1e20
            for i in range(indexes[img], indexes[img+1]): # for i in range((img // 5) * 5, (img // 5) * 5 + 5):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            # ranks array with best rank achieved for index image (among 5 ground-truth captions if coco)
            ranks[img] = rank
            # array with most similar caption retrieved for index image
            top1[img] = inds[0]
    elif mode is 't2i':
        for img in trange(npts):
            # Get query captions
            caps = captions[indexes[img]:indexes[img+1]]
            d = ims.mm(caps.t()).cpu().numpy().T
            inds = np.zeros(d.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[indexes[img] + i] = np.where(inds[i] == img)[0][0]
                top1[indexes[img] + i] = inds[i][0]
    else:
        raise ValueError('mode not correct')

    # Compute metrics recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr
		

def recall_test(img_embs, cap_embs, tot_lengths, model):
    with torch.no_grad():
        # caption retrieval
        (r1, r5, r10, medr, meanr) = recall(img_embs, cap_embs, model, mode='i2t', lenghts=tot_lengths)
        # print("OUR Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = recall(img_embs, cap_embs, model, mode='t2i', lenghts=tot_lengths)
        # print("OUR Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        # print('Sum score: %.1f' % currscore)

    return r1, r5, r10, r1i, r5i, r10i, currscore


# Questa è la funzione da chiamare con tutti gli image embedding e caption embedding estratti da un modello sul val o test 
def recall_1k_5fold_test(img_embs, cap_embs, tot_lengths=None, model=None):
    with torch.no_grad():
        r1_5fold = list()
        r5_5fold = list()
        r10_5fold = list()
        r1i_5fold = list()
        r5i_5fold = list()
        r10i_5fold = list()

        img_embs = torch.split(img_embs, 5000, dim=0)
        cap_embs = torch.split(cap_embs, 5000, dim=0)
        # tot_lengths = torch.split(tot_lengths, 5000, dim=0)

        for i in range(5):
            # r1, r5, r10, r1i, r5i, r10i, _ = recall_test(img_embs[i * 5000:(i + 1) * 5000],
            #                                              cap_embs[i * 5000:(i + 1) * 5000],
            #                                              tot_lengths[i * 5000:(i + 1) * 5000], model)
            print('Computing Test recall... chunk %s of 5' % (i+1))
            r1, r5, r10, r1i, r5i, r10i, _ = recall_test(img_embs[i].to(device), cap_embs[i].to(device),
                                                         None, None)

            r1_5fold.append(r1)
            r5_5fold.append(r5)
            r10_5fold.append(r10)
            r1i_5fold.append(r1i)
            r5i_5fold.append(r5i)
            r10i_5fold.append(r10i)

        r1 = float(np.mean(r1_5fold))
        r5 = float(np.mean(r5_5fold))
        r10 = float(np.mean(r10_5fold))
        r1i = float(np.mean(r1i_5fold))
        r5i = float(np.mean(r5i_5fold))
        r10i = float(np.mean(r10i_5fold))
        rsum = r1 + r5 + r10 + r1i + r5i + r10i

        print("Test 1K 5Folds - Recall Image to text: %.2f, %.2f, %.2f" % (r1, r5, r10))
        print("Test 1K 5Folds - Recall Text to image: %.2f, %.2f, %.2f" % (r1i, r5i, r10i))
        print('Test 1K 5Folds - Sum score: %.2f' % rsum)

    return r1, r5, r10, r1i, r5i, r10i, rsum


def compute_recall(img_embs, cap_embs, tot_lengths=None, model=None):
    with torch.no_grad():
        # tot_lengths = torch.split(tot_lengths, 5000, dim=0)

        # r1, r5, r10, r1i, r5i, r10i, _ = recall_test(img_embs[i * 5000:(i + 1) * 5000],
        #                                              cap_embs[i * 5000:(i + 1) * 5000],
        #                                              tot_lengths[i * 5000:(i + 1) * 5000], model)
        # print('Computing Test recall...')
        r1, r5, r10, r1i, r5i, r10i, _ = recall_test(img_embs.to(device), cap_embs.to(device),
                                                     None, None)
        rsum = r1 + r5 + r10 + r1i + r5i + r10i

        print("Recall Image to text: %.2f, %.2f, %.2f" % (r1, r5, r10))
        print("Recall Text to image: %.2f, %.2f, %.2f" % (r1i, r5i, r10i))
        print('Sum score: %.2f' % rsum)

    return r1, r5, r10, r1i, r5i, r10i, rsum