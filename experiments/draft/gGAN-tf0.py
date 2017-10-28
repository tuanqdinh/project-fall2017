
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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

    def plot(self, samples):
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig

    def build_model(self, mnist, batch_size):
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

        D_solver = tf.train.AdamOptimizer().minimize(D_loss,
                                                        var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss,
                                                        var_list=self.theta_G)
        # Training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out/'):
            os.makedirs('out/')
        nIters = 10000
        miniIters = 1000
        i = 0
        for it in range(nIters):
            if it % miniIters == 0:
                samples = sess.run(G_sample,
                           feed_dict={self.Z: self.sample_Z(16, self.dim_z)})
                fig = self.plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)),
                           bbox__hnches='tight')
                i += 1
                plt.close(fig)

            X_mb, _ = mnist.train.next_batch(batch_size)
            _, D_loss_curr = sess.run([D_solver, D_loss],
                    feed_dict={self.X: X_mb,
                                self.Z: self.sample_Z(batch_size, self.dim_z)})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                    feed_dict={self.Z: self.sample_Z(batch_size, self.dim_z)})

            if it % miniIters == 0:
               print('Iter: {}'.format(it))
               print('D loss: {:.4}'. format(D_loss_curr))
               print('G_loss: {:.4}'.format(G_loss_curr))

# end class

if __name__ == '__main__':
    mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
    dim_z = 100
    dim_x = mnist.train.images.shape[1]
    dim_y = mnist.train.labels.shape[1]
    dim_h = 128
    batch_size = 64

    gg = GGAN(dim_z, dim_x, dim_h)
    gg.build_model(mnist, batch_size)




# ### Evaluation
