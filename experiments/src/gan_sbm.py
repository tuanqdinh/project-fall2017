import tensorflow as tf
import numpy as np
import os
import time
from __init__ import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GGAN:
    def __init__(self, im_size, model_name):
        self.im_size = im_size # image size
        self.dim_z = 100 # Noise size
        self.dim_h = 128 # hidden layers
        self.dim_x_2 =  int(im_size*(im_size + 1)/2)
        self.dim_x = im_size**2 # Real input Size
        self.model_name = model_name

        # Generator network params
        initializer = tf.contrib.layers.xavier_initializer()
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.G_W1 = tf.Variable(initializer(shape=[self.dim_z, self.dim_h]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))

        self.G_W2 = tf.Variable(initializer([self.dim_h, self.dim_x_2]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.dim_x_2]))

        self.G_W3 = tf.Variable(initializer([self.dim_x_2, self.dim_x]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.dim_x]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3, \
                    self.G_b1, self.G_b2, self.G_b3]

        # Discriminator network params
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
        self.D_W1 = tf.Variable(initializer([self.dim_x, self.dim_h]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.D_W2 = tf.Variable(initializer([self.dim_h, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def generator(self, z):
        pkeep = 0.5
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        # G_h1 = tf.nn.dropout(G_h1, pkeep)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        # G_h2 = tf.nn.dropout(G_h2, pkeep)
        G_h3 = tf.nn.tanh(tf.matmul(G_h2, self.G_W3) + self.G_b3)
        # G_prob = tf.sign(G_h3)

        G_flat = tf.reshape(G_h3, [-1, self.dim_x])

        return G_flat

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def sample_z(self, m, n):
        # sample from a gaussian distribution
        # return np.random.normal(size=[m, n], loc = 0, scale = 1)
        return np.random.uniform(-1., 1., size=[m, n])

    def build_model(self, data, batch_size, n_epochs, print_counter,
                            inp_path, out_path, n_figs):
        G_sample = self.generator(self.Z)
        D_real, D_logit_real = self.discriminator(self.X)
        D_fake, D_logit_fake = self.discriminator(G_sample)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_real,
                # labels=tf.random_uniform(tf.shape(D_logit_real), 0.7, 1)))
                labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake,
                # labels=tf.random_uniform(tf.shape(D_logit_fake), 0, 0.3)))
                labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake

        G_real = tf.reshape(G_sample, [-1, 28, 28])
        G_t = tf.transpose(G_real, perm=[0, 2, 1])
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))) \
      + 0.1 * tf.reduce_mean(tf.norm(G_real - G_t, axis=(1,2), ord='fro'))
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.inverse_time_decay(0.9, global_step, \
                                        # 1, 1)
        D_solver = tf.train.GradientDescentOptimizer(0.01).minimize(D_loss,
                                                        var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss,
                                                        var_list=self.theta_G)
        # Training
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)

        tic = time.clock()
        dataset_size = data.shape[0]
        for it in range(n_epochs):
            batch_samp = np.random.randint(dataset_size, size=batch_size)
            X_raw = data[batch_samp, :]
            # X_mb = self.im_sizeormalize_image(X_raw)
            X_mb = X_raw
            if it % print_counter == 0:
                idx = it // print_counter
                # Training set
                # from IPython import embed; embed()
                plot(X_mb[:n_figs,:], self.im_size, inp_path, idx)
                # Synthetic samples
                samples = sess.run(G_sample,
                       feed_dict={self.Z: self.sample_z(n_figs, self.dim_z)})
                # samples = (samples + 1) / 2.0 #denormalize
                plot(samples[:n_figs,:], self.im_size, out_path, idx)

            _, D_loss_curr= sess.run([D_solver, D_loss], feed_dict={
                self.X: X_mb, self.Z: self.sample_z(batch_size, self.dim_z)})
            _, G_loss_curr= sess.run([G_solver, G_loss], feed_dict={
                        self.Z: self.sample_z(batch_size, self.dim_z)})
            if it % print_counter == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
        toc = time.clock()
        print('Time for training: {}'.format(toc-tic))
        saver = tf.train.Saver()
        saver.save(sess, self.model_name)

    def generate_sample(self, m):
        # use the former session
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.model_name)
        G_sample = self.generator(self.Z)
        samples = sess.run(G_sample,
                 feed_dict={self.Z: self.sample_z(m, self.dim_z)})
        return samples
# end class
