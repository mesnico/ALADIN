import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pickle
import math

t2i_h5 = 'alad_features_ann_v3c1_t2i.h5'
n_bins = 150

def load_similarities(h5_file):
    with h5py.File(h5_file, 'r') as f:
        dists = f['distances'][:]
        sims = 1 - dists # cosine sims
        sims = sims.flatten()
    return sims

if __name__ == '__main__':
    colors = ['blue', 'red']
    data = {'t2i':       load_similarities(t2i_h5)}

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    # close = 'purple'
    ax.grid(True)
    # colors = [colors_palette[i] for i in data['color_ids']]
    # labels = [legend[i] for i in data['color_ids']]

    for i, (name, d) in enumerate(data.items()):
        ax.hist(d, bins=n_bins, density=True, alpha=0.7, label=name)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Norm. Frequency')
    # ax.set_title('Volume and percent change')
    ax.legend(loc="upper right")
    fig.tight_layout()

    # plt.savefig('i2i_t2i_distributions.pdf')
    plt.show()