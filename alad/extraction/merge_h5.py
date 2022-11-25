import h5py
import glob
import numpy as np
import tqdm

path = '/media/nicola/SSD/VBS_Features/V3C1_ALADIN'
out_file = '/media/nicola/SSD/VBS_Features/aladin_v3c1_features.h5'
img_list = '/media/datino/Dataset/VBS/V3C_dataset/V3C1/v3c1_image_list.txt'

# get original number of images
with open(img_list, 'r') as f:
    num_images = len(f.read().splitlines())

print(f'Original number of images: {num_images}')

with h5py.File(out_file, mode='w') as h5fw:
    row1 = 0
    for h5name in tqdm.tqdm(glob.glob(path + '/*.h5')):
        # open file in read mode and open the datasets
        h5fr = h5py.File(h5name,'r') 
        features = h5fr['features'][:]
        img_names = h5fr['image_names'][:]
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
        # else:
        #     h5fw['features'].resize( (row1+dslen, feat_dim) )
        #     h5fw['image_names'].resize( (row1+dslen, ) )
        #     h5fw['features'][row1:row1+dslen,:] = features[:]
        #     h5fw['image_names'][row1:row1+dslen] = img_names[:]
        row1 += dslen

    print(f'Num total features: {row1}')
    assert row1 == num_images