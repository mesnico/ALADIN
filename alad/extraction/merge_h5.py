import h5py
import glob
import os
import numpy as np
import tqdm
import argparse

def main(args):
    # get original number of images
    with open(args.img_list, 'r') as f:
        num_images = len(f.read().splitlines())

    print(f'Original number of images: {num_images}')

    already_existing = os.path.exists(args.out_file)
    mode = 'r' if already_existing and not args.force else 'w'
    with h5py.File(args.out_file, mode=mode) as h5fw:
        if mode == 'w':
            print(f'Writing on file {args.out_file}')
            row1 = 0
            files = glob.glob(args.path + '/*.h5')
            ordered_files = sorted(files, key=lambda x: int(x.split('_')[-2]))

            for i, h5name in enumerate(tqdm.tqdm(ordered_files)):
                # open file in read mode and open the datasets
                h5fr = h5py.File(h5name,'r')
                features = h5fr['features'][:]
                img_names = h5fr['image_names'][:]
                if i == 0:
                    bs = len(features)
                dslen = features.shape[0]
                feat_dim = features.shape[1]
                if row1 == 0: 
                    # create the new datasets in the new file
                    feats = h5fw.create_dataset("features", (num_images, feat_dim), dtype=np.float32)
                    dt = h5py.string_dtype(encoding='utf-8')
                    img_ids = h5fw.create_dataset('image_names', (num_images, ), dtype=dt)
                # append data
                h5fw['features'][row1:row1+dslen,:] = features[:]
                h5fw['image_names'][row1:row1+dslen] = img_names[:]

                row1 += dslen
        else:
            row1 = len(h5fw['features'])
            ids = h5fw['image_names']
            bs = len(h5py.File(glob.glob(args.path + '/*.h5')[0], 'r')['features'])

        print(f'Num total features: {row1}')
        assert row1 == num_images

        # check duplicates
        ids = h5fw['image_names'][:]
        ids = [id.decode() for id in ids]
        duplicates = len(ids) - len(set(ids))
        print(f'{duplicates} duplicates found.')
        if duplicates > 0:
            seen = set()
            dupes_idx, dupes = zip(*[(i, x) for i, x in enumerate(ids) if x in seen or seen.add(x)])
            possibly_corrupted_files = [(did // bs) * bs for did in dupes_idx]
            possibly_corrupted_files = set(possibly_corrupted_files)
            possibly_corrupted_files = [f'{s}-{(s + bs)}' for s in possibly_corrupted_files]
            print(f'Possibly corrupted files: {possibly_corrupted_files}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path where h5 files are stored")
    parser.add_argument("--out_file", type=str, required=True, help="path of the output h5 file")
    parser.add_argument("--img_list", type=str, required=True, help="path to the image list to use as a reference to check if all the images are present.")
    parser.add_argument("--force", action="store_true", help="whether to force the overwrite of the output h5 file, if it already exists")
    args = parser.parse_args()

    main(args)