import numpy as np
import os
from gan_sbm import GGAN
from __init__ import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_mnist(gg):
    batch_size = 32
    nIters = 10000
    print_counter = 1000
    X_dim = 784
    z_dim = 10
    h_dim = 128
    N=28
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    gg = GGAN(N, True)
    gg.build_model(mnist, batch_size, nIters, print_counter)

if __name__ == '__main__':
    im_size = 28
    n_clusters = 4
    n_fig_unit = 2
    # generate and dump graphs
    trainning = True
    testing = True
    model_name = '../models/gan_sbm_4p1.ckpt'

    generating = False
    filename = '../dataset/sbm_4p1.npy'
    if not(generating):
        data = np.load(filename)
    else:
        dataset_size = 100000
        prob_in = 0.9
        prob_out = 0.1
        perm = 2
        data = generate_sbm_data(dataset_size, im_size, n_clusters,
                    prob_in, prob_out, perm)
        np.save(filename, data)

    gg = GGAN(im_size, model_name)
    if trainning:
        batch_size = 64
        n_epochs = 100000
        print_counter = 10000
        inp_path = os.path.abspath('../out_samples/inp_training')
        out_path = os.path.abspath('../out_samples/out_gen')
        gg.build_model(data, batch_size, n_epochs, print_counter,
                    inp_path, out_path, n_fig_unit ** 2)
    # Testing
    if testing:
        samples = gg.generate_sample(200)
        out_path = os.path.abspath('../out_clustering/out_gen')
        inp_path = os.path.abspath('../out_clustering/inp_training')
        reconstruct_image(samples, out_path, True, im_size, n_clusters,
                                 n_fig_unit, 10)
        reconstruct_image(data, inp_path, True, im_size, n_clusters,
                                 n_fig_unit, 10)
