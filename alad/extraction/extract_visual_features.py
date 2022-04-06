from __future__ import absolute_import, division, print_function
import tqdm
import h5py

import torch
import numpy as np
from alad.extraction.retrieval_utils import load_oscar

def main():
    args, student_model, test_loader = load_oscar()
    extract_features(args, student_model, test_loader)

def extract_features(args, model, data_loader):
    # switch to evaluate mode
    model.eval()

    with h5py.File(args.features_h5, 'w') as f:
        feats = f.create_dataset("features", (len(data_loader.dataset), 768), dtype=np.float32)
        dt = h5py.string_dtype(encoding='utf-8')
        img_ids = f.create_dataset('image_names', (len(data_loader.dataset), ), dtype=dt)

        for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
            dataset_idxs, image_names, example_imgs = batch_data
            dataset_idxs = list(dataset_idxs)

            # compute the embeddings and save in hdf5
            with torch.no_grad():
                img_cross_attention, _, _, _, img_length, _, _ = model.forward_emb(example_imgs, None)
                feats[dataset_idxs, :] = img_cross_attention.cpu().numpy()
                for image_name, idx in zip(image_names, dataset_idxs):
                    img_ids[idx] = np.array(image_name.encode("utf-8"), dtype=dt)


if __name__ == "__main__":
    main()