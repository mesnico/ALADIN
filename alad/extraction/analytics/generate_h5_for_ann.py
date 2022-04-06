import random

import h5py
import tqdm
import numpy as np
import argparse
import dask.array as da
import os
import pickle
from alad.extraction.image_retrieval import QueryEncoder
from alad.extraction.retrieval_utils import compute_if_dask

captions_txt = '/media/nicola/Data/Workspace/TERE/vbs/ann_experiments/vbs_log_captions.txt'
tern_checkpoint = "runs/ALIGN/WITH_ICLS/tere_alignment_tern_teran_uncertainty_weighting/model_best_ndcgspice.pth.tar"
feat_dbs = {'v3c1': "alad_features.h5"}
            # 'v3c2': "/media/datino/Dataset/VBS/descriptors_feature_extracted/TERN/V3C2/tern_features.h5"}

random.seed(42)

def get_caption_features():
    with open(captions_txt, 'r') as f:
        captions = f.read().splitlines()

    # need to forward the network for that...
    cap_features = []
    qe = QueryEncoder()
    for cap in tqdm.tqdm(captions):
        cap_feature = qe.get_text_embedding(cap)
        cap_features.append(cap_feature)
    cap_features = np.stack(cap_features)

    return cap_features

def get_image_features(split):
    data = h5py.File(feat_dbs[split], 'r')
    features = data['features']
    return features

def similarity_search(query_features, items_features, num_neighbors=100):
    all_neighbors = np.zeros((query_features.shape[0], num_neighbors), dtype=np.int)
    all_distances = np.zeros((query_features.shape[0], num_neighbors), dtype=np.float32)

    items_features = da.from_array(items_features, chunks=(10000, 1024))

    # returns distances, neighbors for each query
    for i in tqdm.trange(query_features.shape[0]):
        query_feat = query_features[i]
        similarities = items_features.dot(query_feat)
        similarities = compute_if_dask(similarities, progress=False)
        distances = 1 - similarities

        # sort by decreasing similarities
        ordered_neighbors = np.argsort(distances).astype(int)[:num_neighbors]
        ordered_distances = distances[ordered_neighbors]
        all_neighbors[i] = ordered_neighbors
        all_distances[i] = ordered_distances
        # if i == 3:
        #     break
    # all_neighbors = np.stack(all_neighbors)
    # all_distances = np.stack(all_distances)

    return all_neighbors, all_distances

def get_random_queries_from_items(item_features, how_many=5000):
    cached_indexes_file = 'vbs/ann_experiments/cached_queries_indexes.pkl'
    if not os.path.exists(cached_indexes_file):
        indices = random.sample(range(item_features.shape[0]), how_many)
        indices = sorted(indices)
        with open(cached_indexes_file, 'wb') as f:
            pickle.dump(indices, f)
    else:
        with open(cached_indexes_file, 'rb') as f:
            indices = pickle.load(f)

    random_queries = item_features[indices]
    return random_queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default='v3c1', help="v3c1 or v3c2")
    parser.add_argument("--type", type=str, default='t2i', help="t2i or i2i")
    parser.add_argument("--num_neighbors", type=int, default=100, help="number of neighbors to consider for each query")
    args = parser.parse_args()

    num_neighbors = args.num_neighbors + 1 if args.type == 'i2i' else args.num_neighbors

    if args.type == 't2i':
        query_features = get_caption_features()
        item_features = get_image_features(args.split)
    elif args.type == 'i2i':
        item_features = get_image_features(args.split)
        query_features = get_random_queries_from_items(item_features, how_many=5000)
    else:
        raise NotImplementedError

    neighbors, distances = similarity_search(query_features, item_features, num_neighbors=num_neighbors)

    if args.type == 'i2i':
        # remove the first element(the query itself) from the resultsets
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]

    # save to hdf5
    out_hdf5_filename = 'alad_features_ann_{}_{}.h5'.format(args.split, args.type)
    print('Writing on {}'.format(out_hdf5_filename))
    with h5py.File(out_hdf5_filename, 'w') as f:
        f.create_dataset('train', data=item_features)
        f.create_dataset('test', data=query_features)
        f.create_dataset('neighbors', data=neighbors)
        f.create_dataset('distances', data=distances)