import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import SpectralClustering

# TF functions
# def xavier_init(self, size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)


def plot(samples, im_size, path, idx, n_fig_unit=2):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(n_fig_unit, n_fig_unit)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(im_size, im_size), cmap='Greys_r')

    plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
                bbox__hnches='tight')
    plt.close(fig)
    return fig


def normalize_image2(x):
    x_new = x / np.linalg.norm(x, 'fro')
    return x_new


def normalize_image(x, a, b):
    m = np.min(x)
    i_range = np.max(x) - m
    x_new = (x - m) * (b - a) / i_range + a
    return x_new


def get_edge(x, p):
    if x >= p:
        return 1
    else:
        return -1


def generate_sbm_matrix(im_size, n_clusters, prob_in, prob_out, perm, col_ord):
    step = int(im_size / n_clusters)
    rand_noise = np.random.rand(im_size, im_size)
    a_matrix = (rand_noise + np.transpose(rand_noise)) / 2.0
    # Create community

    for k in range(n_clusters):
        start = k * step
        end = (k + 1) * step
        for r in range(start, end):
            # community
            for c in range(start, end):
                a_matrix[r, c] = get_edge(a_matrix[r, c], 1 - prob_in)
            # non community
            non_com_set = [x for x in range(im_size)
                           if x not in range(start, end)]
            for l in non_com_set:
                a_matrix[r, l] = get_edge(a_matrix[r, l], 1 - prob_out)
    for i in range(im_size):
        a_matrix[i, i] = -1

    if perm != 0:
        if perm == -1:
            col_ord = np.arange(im_size)
            np.random.shuffle(col_ord)
        a_matrix = a_matrix[:, col_ord]
        a_matrix = a_matrix[col_ord, :]

    a_flat = np.reshape(a_matrix, im_size**2)
    return a_flat


def generate_sbm_data(m, im_size, n_clusters, prob_in, prob_out, perm):
    graphs = np.zeros((m, im_size * im_size))
    col_ords = []
    if perm > 0:
        for k in range(perm):
            col_ord = np.arange(im_size)
            np.random.shuffle(col_ord)
            col_ords.append(col_ord)

    for k in range(m):
        if perm > 0:
            col_ord = col_ords[k % perm]
        else:
            col_ord = []
        graphs[k, :] = generate_sbm_matrix(im_size,
                                           n_clusters, prob_in, prob_out, perm, col_ord)
    return graphs


def spec_cluster(adj_matrix, im_size, n_clusters, normalized):
    #normalize
    adj_matrix = normalize_image(adj_matrix, 0, 1)

    im = np.reshape(adj_matrix, (im_size, im_size))
    sc = SpectralClustering(n_clusters, affinity='precomputed')
    sc.fit(im)
    inds = np.argsort(sc.labels_)
    sorted_im = im[inds, :]  # row
    sorted_im = sorted_im[:, inds]  # columns
    return sorted_im


def reconstruct_image(data, out_path, normalized, im_size, n_clusters,
                      n_fig_unit=2, n_test=10):
    # assure()
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    n_figs = n_fig_unit**2
    for i in range(n_test):
        start = i * n_figs
        end = (i + 1) * n_figs
        collection = np.zeros((n_figs, im_size * im_size))
        for k in range(start, end):
            reconstructed_im = spec_cluster(data[k],
                                            im_size, n_clusters, normalized)
            idx = k % n_figs
            collection[idx, :] = np.reshape(reconstructed_im, im_size**2)
        plot(collection, im_size, out_path, i, n_fig_unit)
