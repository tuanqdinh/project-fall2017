import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import networkx as nx

class GGAN:
    def __init__(self, dim_z, dim_x, dim_h):
        self.dim_z = dim_z # Noise size
        self.dim_x = dim_x # Real input Size
        self.dim_h = dim_h # hidden layers

        # Generator network params
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.G_W1 = tf.Variable(self.xavier_init([self.dim_z, self.dim_h]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.G_W2 = tf.Variable(self.xavier_init([self.dim_h, self.dim_x]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.dim_x]))
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # Discriminator network params
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
        self.D_W1 = tf.Variable(self.xavier_init([self.dim_x, self.dim_h]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.D_W2 = tf.Variable(self.xavier_init([self.dim_h, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        self.c = 0
        self.lr = 1e-3 # learning rate

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def plot(self, samples, N):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(N, N), cmap='Greys_r')
        return fig

    def build_model(self, data, batch_size):
        G_sample = self.generator(self.Z)
        D_real, D_logit_real = self.discriminator(self.X)
        D_fake, D_logit_fake = self.discriminator(G_sample)

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.inverse_time_decay(0.9, global_step, \
                                        # 1, 1)
        D_solver = tf.train.GradientDescentOptimizer(0.9).minimize(D_loss,
                                                        var_list=self.theta_D)
        G_solver = tf.train.GradientDescentOptimizer(0.9).minimize(G_loss,
                                                        var_list=self.theta_G)
        # Training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out_gen/'):
            os.makedirs('out_gen/')
        if not os.path.exists('inp_training/'):
            os.makedirs('inp_training/')

        nIters = 100000
        print_counter = 1000
        i = 0
        j = 0
        tic = time.clock()
        dataset_size = data.shape[0]

        for it in range(nIters):
            if it % print_counter == 0:
                samples = sess.run(G_sample,
                           feed_dict={self.Z: self.sample_Z(16, self.dim_z)})
                fig = self.plot(samples, N)
                plt.savefig('out_gen/{}.png'.format(str(i).zfill(3)),
                            bbox__hnches='tight')
                i += 1
                plt.close(fig)

            batch_samp = np.random.randint(dataset_size, size=batch_size)
            X_mb = data[batch_samp, :]
            if it % print_counter == 0:
                fig = self.plot(X_mb[:16, ], N)
                plt.savefig('inp_training/{}.png'.format(str(j).zfill(3)),
                            bbox__hnches='tight')
                j += 1
                plt.close(fig)

            _, D_loss_curr = sess.run([D_solver, D_loss],
                    feed_dict={self.X: X_mb,
                                self.Z: self.sample_Z(batch_size, self.dim_z)})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                    feed_dict={self.Z: self.sample_Z(batch_size, self.dim_z)})

            if it % print_counter == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()

        toc = time.clock()
        print('Time for training: {}'.format(toc-tic))

        # Test the generator samples
        # nsamples_gen_test = 100;
        # samples_clustering_coeff = np.zeros(nsamples_gen_test)
        # samples = sess.run(G_sample,
        #     feed_dict={self.Z: self.sample_Z(nsamples_gen_test, self.dim_z)})
        # for it in range(nsamples_gen_test):
        #     samp_adj_mat = np.reshape(samples[it, :], (N, N))
        #     samp_G = nx.from_numpy_matrix(samp_adj_mat)
        #     samples_clustering_coeff[it] = nx.average_clustering(samp_G)
        # print('samples_clustering_coeff:',samples_clustering_coeff)
        #
        # nsamples_real_test = 10
        # # gs = self.generate_sbm(N, nsamples_real_test)
        # gs = np.random.rand(nsamples_real_test, N, N)
        # for it in range(nsamples_real_test):
        #     samp_adj_mat = np.reshape(samples[it, :], (N, N))
        #     samp_G = nx.from_numpy_matrix(samp_adj_mat)
        #     samples_clustering_coeff[it] = nx.average_clustering(samp_G)
        #
        # print('real_clustering_coeff:',samples_clustering_coeff)
        # from IPython import embed; embed()

# end class
def set_edge(p, x):
    if x < p:
        return 0

def generate_matrix(N, n_unit, pDist):
    step = int(N / n_unit)
    a = np.random.rand(N, N)
    for c in range(n_unit):
        for r in range(n_unit):
            p = r * n_unit + c
            for i in range(step * r, step * (r + 1)):
                for j in range(step * c, step * (c + 1)):
                    a[i, j] = set_edge(pDist[p], a[i, j])
    # np.random.shuffle(a)
    # col_ord = np.random.shuffle(np.arange(N))
    # a = a[:, col_ord]
    a_flat = np.reshape(a, N * N)
    return a_flat

def generate_sbm(N, m):
    n_unit = 2
    # n_clusters = n_unit**2
    # pDist = np.arange(1, n_clusters + 1) / sum(np.arange(1, n_clusters + 1))
    pDist = [0.99, 0.01, 0.01, 0.99]
    graphs = np.zeros((m, N*N))
    for k in range(m):
        graphs[k, :] = generate_matrix(N, n_unit, pDist)

    return graphs

if __name__ == '__main__':
    N = 90
    dim_z = 500
    dim_x = N * N
    dim_h = 512
    dataset_size = 10000

    # generate and dump graphs
    is_stored = True
    filename = 'matrices.npy'
    if is_stored:
        data = np.load(filename)
    else:
        data = generate_sbm(N, dataset_size)
        np.save(filename, data)

    #Training right now is very sensitive to the batch size. This might
    #be a fundamental/theoretical/conceptual issue.
    # If batch size is 1 => good, 10 => bad, or maybe learn faster => white
    batch_size = int(0.0001 * dataset_size)

    gg = GGAN(dim_z, dim_x, dim_h)
    gg.build_model(data, batch_size)




# ### Evaluation
