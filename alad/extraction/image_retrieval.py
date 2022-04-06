import os
import time

import h5py
import torch
import dask.array as da
import numpy as np
from alad.extraction.retrieval_utils import compute_if_dask, scalar_quantization, load_oscar


class QueryEncoder:
    def __init__(self):
        args, student_model, dataloader = load_oscar()
        self.model = student_model
        self.model.eval()

        self.dataset = dataloader.dataset

    def get_text_embedding(self, caption):
        examples_text = [self.dataset.tensorize_example_disentangled(text_a=caption, img_feat=None, text_b=None, return_lengths=True)]
        examples_text = [torch.stack(t) if isinstance(t[0], torch.Tensor) else t for t in zip(*examples_text)]
        with torch.no_grad():
            _, txt_feature, _, _, _, _, _ = self.model.forward_emb(None, examples_text)
            txt_feature = txt_feature.cpu().squeeze(0).numpy()
            return txt_feature


# TODO scalar quantization
class ImageRetrieval:
    def __init__(self, features_h5_filename, use_dask=True):
        # Initialize image features
        image_data = h5py.File(features_h5_filename, 'r')

        self.features = da.from_array(image_data['features'], chunks=(10000, 512)) if use_dask else image_data['features'][:]
        self.shot_ids = image_data['image_names']

        self.rotation_matrix = None

    def encode_query_and_postprocess(self, query_feat, enable_scalar_quantization=False, factor=0, thr=30):
        if enable_scalar_quantization:
            print('Using scalar quantization')
            query_feat = scalar_quantization(query_feat, threshold=thr, factor=factor,
                                             rotation_matrix=self.rotation_matrix, subtract_mean=False)
            non_zero_elems = len(np.nonzero(query_feat)[0])
            print('Number of non-zero elements in query: {}'.format(non_zero_elems))
            print('Max query vector value: {}'.format(max(query_feat)))
        else:
            non_zero_elems = len(np.nonzero(query_feat)[0])

        return query_feat, non_zero_elems

    def encode_features_and_postprocess(self, enable_scalar_quantization=False, factor=0, thr=30):
        if enable_scalar_quantization:
            features = scalar_quantization(self.features, threshold=thr, factor=factor,
                                           rotation_matrix=self.rotation_matrix, subtract_mean=True)
        else:
            features = self.features
        ids = self.shot_ids

        return features, ids

    def sequential_search(self, query_feat, enable_scalar_quantization=False, factor=0, thr=30):
        # given a caption as a query, search the most relevant images and return their indexes
        cap_emb_aggr, non_zero_elems = self.encode_query_and_postprocess(query_feat, enable_scalar_quantization, factor, thr)
        features, _ = self.encode_features_and_postprocess(enable_scalar_quantization, factor, thr)

        # similarities = np.zeros(len(self.features))
        # img_ids = []
        # for i, (k, v) in enumerate(tqdm.tqdm(self.features.items())):
        #     if enable_scalar_quantization:
        #         v = scalar_quantization(v, rotation_matrix=self.rotation_matrix, subtract_mean=True)
        #
        #     similarities[i] = np.dot(cap_emb_aggr, v)
        #     img_ids.append(k)

        # use the power of dask
        print('Computing similarities...')
        similarities = features.dot(cap_emb_aggr)
        similarities = compute_if_dask(similarities)

        # sort by decreasing similarities
        inds = np.argsort(similarities)[::-1].astype(int)
        img_ids = [self.shot_ids[i].decode() for i in inds]
        ordered_similarities = similarities[inds]

        return img_ids, ordered_similarities, non_zero_elems

if __name__ == '__main__':

    query_encoder = QueryEncoder()
    vr = ImageRetrieval('alad_features.h5', use_dask=False)
    text = 'A man and a woman are talking' #'A man is doing acrobatics on a bike'

    # text = input('Text query: ')
    start_time = time.time()
    query_feat = query_encoder.get_text_embedding(text)
    query_time = time.time()
    img_ids, similarities, _ = vr.sequential_search(query_feat)
    end_time = time.time()
    print(img_ids[:20])
    print('Query encoding time: {}; Search time: {}'.format(query_time - start_time, end_time - query_time))