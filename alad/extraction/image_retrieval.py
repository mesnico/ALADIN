import os
import time
import faiss

import h5py
import torch
import dask.array as da
import numpy as np
import tqdm
from alad.extraction.retrieval_utils import load_oscar
import surrogate


class QueryEncoder:
    def __init__(self, str_args):
        args, student_model, dataloader = load_oscar(str_args)
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


class ImageRetrieval:
    def __init__(self, features_h5_filenames, sq_threshold=None, sq_factor=100, batch_size=10000, sq_method='thr'):
        self.faiss = sq_threshold is None
        assert sq_method in ['thr', 'topk']

        if isinstance(features_h5_filenames, str):
            features_h5_filenames = [features_h5_filenames]

        d = 768

        for i, h5_file in enumerate(tqdm.tqdm(features_h5_filenames)):
            image_data = h5py.File(h5_file, 'r')
            features = image_data['features'][:]

            if i == 0:
                self.shot_ids = image_data['image_names'][:]

                # create the indexes
                if not self.faiss:
                    rotation_matrix = np.load('alad/extraction/random_rotation_matrix.npy').astype(np.float32)
                    if sq_method == 'thr':
                        self.index = surrogate.ThresholdSQ(
                            d,
                            threshold_percentile=sq_threshold,
                            sq_factor=sq_factor,
                            subtract_mean=False,
                            l2_normalize=False,
                            rotation_matrix=rotation_matrix,
                            parallel=False
                        )
                    else:
                        self.index = surrogate.TopKSQ(
                            d,
                            keep=1 - (sq_threshold / 100.0),
                            sq_factor=sq_factor,
                            l2_normalize=False,
                            rotation_matrix=rotation_matrix,
                            parallel=False
                        )
                    self.index.train(features[:500000])
                else:
                    metric = faiss.METRIC_INNER_PRODUCT
                    self.index = faiss.index_factory(d, 'Flat', metric)
            else:
                # features = np.concatenate((features, image_data['features'][:]), axis=0)
                self.shot_ids = np.concatenate((self.shot_ids, image_data['image_names'][:]), axis=0)

            # add items to the index, procedurally
            for i in tqdm.trange(0, len(features), batch_size, desc='ADD'):
                self.index.add(features[i:i+batch_size])

            if not self.faiss:
                # commit
                self.index.commit()

    def get_density(self):
        if self.faiss:
            return None
        else:
            return self.index.density

    def get_number_nz_elems(self):
        if self.faiss:
            return None
        else:
            nz_elems = self.index.db.count_nonzero() / self.index.db.shape[1]
            return nz_elems

    def search(self, q, k=1000):
        # given a caption as a query, search the most relevant images and return their indexes
        q = q[np.newaxis, :]
        sims, idxs = self.index.search(q, k=k)
        sims = sims[0]
        idxs = idxs[0]
        img_ids = [self.shot_ids[i].decode() for i in idxs]

        return img_ids, sims

if __name__ == '__main__':

    sq_thr = 50
    sq_factor = 100

    query_encoder = QueryEncoder('--data_dir /media/nicola/SSD/OSCAR_Datasets/coco_ir --img_feat_file /media/nicola/Data/Workspace/OSCAR/scene_graph_benchmark/output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv --eval_model_dir /media/nicola/SSD/OSCAR_Datasets/checkpoint-0132780 --max_seq_length 50 --max_img_seq_length 34 --load_checkpoint /media/nicola/Data/Workspace/OSCAR/Oscar/alad/runs/alad-alignment-and-distill/model_best_rsum.pth.tar')
    ir = ImageRetrieval(['/media/nicola/SSD/VBS_Features/aladin_v3c1_features.h5'], sq_threshold=sq_thr, sq_factor=sq_factor)
    text = 'A man and a woman are talking' #'A man is doing acrobatics on a bike'

    # text = input('Text query: ')
    start_time = time.time()
    query_feat = query_encoder.get_text_embedding(text)
    query_time = time.time()
    img_ids, similarities = ir.search(query_feat, k=20)
    end_time = time.time()
    print(f"Non zero elems: {ir.get_avg_nz_elems()}")
    print('Query encoding time: {}; Search time: {}'.format(query_time - start_time, end_time - query_time))
    print(img_ids)