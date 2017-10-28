import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import networkx as nx
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GGAN:
    def __init__(self, dim_z, dim_x, dim_h):
        self.dim_z = dim_z # Noise size
        self.dim_x = dim_x # Real input Size
        self.dim_h = dim_h # hidden layers
        self.N = int(np.sqrt(self.dim_x))

        # Generator network params
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.G_W1 = tf.Variable(self.xavier_init([self.dim_z, self.dim_h]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.G_W2 = tf.Variable(self.xavier_init([self.dim_h, self.dim_x]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.dim_x]))
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # Discriminator network params
        self.D_C1 = 32 # first convolutional layer output depth
        self.D_C2 = 64  # second convolutional layer output depth
        self.D_FC = 1024  # fully connected layer
        self.D_stride = 1  # output is 28x28
        # convolution
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
        self.D_W1 = tf.Variable(tf.truncated_normal(
                    [5, 5, 1, self.D_C1], stddev=0.1))
        self.D_b1 = tf.Variable(tf.truncated_normal([self.D_C1], stddev=0.1))
        self.D_W2 = tf.Variable(tf.truncated_normal(
                    [7, 7, self.D_C1, self.D_C2], stddev=0.1))
        self.D_b2 = tf.Variable(tf.truncated_normal([self.D_C2], stddev=0.1))
        # full layer
        self.D_W3 = tf.Variable(tf.truncated_normal(
                    [10*10*self.D_C2, self.D_FC], stddev=0.1))
        self.D_b3 = tf.Variable(tf.truncated_normal([self.D_FC], stddev=0.1))
        self.D_W4 = tf.Variable(tf.truncated_normal([self.D_FC, 1], stddev=0.1))
        self.D_b4 = tf.Variable(tf.truncated_normal([1], stddev=0.1))

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_W4,
                    self.D_b1, self.D_b2, self.D_b3, self.D_b4]

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def discriminator(self, x):
        k = 2 # max pool filter size
        pooling = [1, k, k, 1]
        pkeep = 0.8
        strides = [1, self.D_stride, self.D_stride, 1]

        X = tf.reshape(x, [-1, self.N, self.N, 1])

        h_c1 = tf.nn.relu(tf.nn.conv2d(X, self.D_W1,
                strides=strides, padding='SAME') + self.D_b1)
        h_p1 = tf.nn.max_pool(h_c1, ksize=pooling,
                strides=pooling, padding='SAME')
        h_c2 = tf.nn.relu(tf.nn.conv2d(h_p1, self.D_W2,
                strides=strides, padding='SAME') + self.D_b2)
        h_p2 = tf.nn.max_pool(h_c2, ksize=pooling,
                strides=pooling, padding='SAME')
        h_flat = tf.reshape(h_p2, shape=[-1, 10 * 10 * self.D_C2])
        h_full = tf.nn.relu(tf.matmul(h_flat, self.D_W3) + self.D_b3)
        h = tf.nn.dropout(h_full, pkeep)
        D_logit = tf.matmul(h, self.D_W4) + self.D_b4
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

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
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.inverse_time_decay(0.9, global_step, \
                                        # 1, 1)
        D_solver = tf.train.AdamOptimizer().minimize(D_loss,
                                                        var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss,
                                                        var_list=self.theta_G)
        # Training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out_gen/'):
            os.makedirs('out_gen/')
        if not os.path.exists('inp_training/'):
            os.makedirs('inp_training/')

        nIters = 10
        print_counter = 1
        i = 0
        j = 0
        tic = time.clock()
        dataset_size = data.shape[0]

        for it in range(nIters):
            #out_gen
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
            #inp_training
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
                # print()

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
    N = 40
    dim_z = 1600
    dim_x = N * N
    dim_h = 512
    dataset_size = 2

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
    # batch_size = int(0.001 * dataset_size)
    batch_size =1

    gg = GGAN(dim_z, dim_x, dim_h)
    gg.build_model(data, batch_size)




# ### Evaluation
