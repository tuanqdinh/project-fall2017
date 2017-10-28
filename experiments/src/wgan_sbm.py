import tensorflow as tf
import numpy as np
import os
import time
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from image_helpers import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GGAN:
    def __init__(self, N, test=False):
        self.N = 30 # image size
        self.dim_z = 100 # Noise size
        self.dim_x = N * N # Real input Size
        self.dim_h = 128 # hidden layers
        self.test = test
        self.model_name = 'models/wgan_sbm.ckpt'

        # Generator network params
        initializer = tf.contrib.layers.xavier_initializer()
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.G_W1 = tf.Variable(initializer(shape=[self.dim_z, self.dim_h]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.G_W2 = tf.Variable(initializer([self.dim_h, self.dim_x]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.dim_x]))
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # Discriminator network params
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
        self.D_W1 = tf.Variable(initializer([self.dim_x, self.dim_h]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.D_W2 = tf.Variable(initializer([self.dim_h, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        return D_logit

    def sample_Z(self, m, n):
        # sample from a gaussian distribution
        # return np.random.normal(size=[m, n], loc = 0, scale = 1)
        return np.random.uniform(-1., 1., size=[m, n])

    def build_model(self, data, batch_size, nIters, print_counter, n_figs = 16):
        lr = 1e-4
        lam = 10
        n_disc = 5
        G_sample = self.generator(self.Z)
        D_real= self.discriminator(self.X)
        D_fake= self.discriminator(G_sample)

        eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
        X_inter = eps * self.X + (1. - eps) * G_sample
        grad = tf.gradients(self.discriminator(X_inter), [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

        D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
        G_loss = -tf.reduce_mean(D_fake)

        D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                    .minimize(D_loss, var_list=self.theta_D))
        G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                    .minimize(G_loss, var_list=self.theta_G))

        # Training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists('out_gen/'):
            os.makedirs('out_gen/')
        if not os.path.exists('inp_training/'):
            os.makedirs('inp_training/')

        tic = time.clock()

        for it in range(nIters):
            for _ in range(n_disc):
                if self.test:
                    X_mb, _ = mnist.train.next_batch(batch_size)
                else:
                    dataset_size = data.shape[0]
                    batch_samp = np.random.randint(dataset_size, size=batch_size)
                    X_mb = data[batch_samp, :]
                _, D_loss_curr= sess.run([D_solver, D_loss], feed_dict={
                    self.X: X_mb, self.Z: self.sample_Z(batch_size, self.dim_z)})
                _, G_loss_curr= sess.run([G_solver, G_loss], feed_dict={
                            self.Z: self.sample_Z(batch_size, self.dim_z)})
            if it % print_counter == 0:
                idx = it // print_counter
                # Training set
                plot(X_mb[:n_figs,:], N, 'inp_training', idx)
                # Synthetic samples
                samples = sess.run(G_sample,
                       feed_dict={self.Z: self.sample_Z(n_figs, self.dim_z)})
                # samples = (samples + 1) / 2.0 #denormalize
                plot(samples, N, 'out_gen', idx)

            if it % print_counter == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
        toc = time.clock()
        print('Time for training: {}'.format(toc-tic))
        if ~self.test:
            saver = tf.train.Saver()
            saver.save(sess, self.model_name)

    def generate_sample(self, m):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.model_name)
        G_sample = self.generator(self.Z)
        samples = sess.run(G_sample,
                 feed_dict={self.Z: self.sample_Z(m, self.dim_z)})
        return samples
# end class








# ### Evaluation
