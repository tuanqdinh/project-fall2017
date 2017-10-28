from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

import sys
sys.path.insert(0, os.path.abspath(".."))
from experiments.libs.image_helpers import reconstruct_image, plot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=28, input_width=28, crop=True,
                 batch_size=64, sample_num=64, output_height=28, output_width=28,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = None
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.data_X = self.load_mnist()
        self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.initializer = tf.contrib.layers.xavier_initializer()

        self.build_model()

    def build_model(self):
        self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.G_real = tf.reshape(self.G, [-1, 28, 28])
        self.G_t = tf.transpose(self.G_real, perm=[0, 2, 1])
        self.g_loss = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        #   + 10 * tf.reduce_mean(tf.norm(self.G_real - self.G_t, axis=(1,2), ord='fro'))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.GradientDescentOptimizer(config.learning_rate)\
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)\
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.normal(-1, 1, size=(self.sample_num, self.z_dim))
        # sample_z = np.random.normal(size=[self.sample_num , self.z_dim],
        # loc = 0, scale = 1)

        # Load dataset
        sample_inputs = self.data_X[0:self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        inp_path = './out_samples/inp_training/'
        out_path = './out_samples/out_gen/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(self.data_X),
                             config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.data_X[idx * config.batch_size:(idx + 1)
                                           * config.batch_size]
                # from IPython import embed; embed()
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                    self.inputs: batch_images,
                    self.z: batch_z
                })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={
                    self.z: batch_z
                })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss
                # does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images
                })
                errG = self.g_loss.eval({
                    self.z: batch_z
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 1000) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: batch_images
                        }
                    )
                    # from IPython import embed; embed()
                    save_images(batch_images, image_manifold_size(batch_images.shape[0]),
                                './out_samples/inp_training/input_{:02d}_{:04d}.png'.format(epoch, idx))

                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './out_samples/out_gen/sample_{:02d}_{:04d}.png'.format(epoch, idx))

                    print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                          (d_loss, g_loss))

                if np.mod(counter, 1000) == 2:
                    self.save(config.checkpoint_dir, counter)

    def generate_sample(self, config):
        import re
        # Generate sample and store here
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print("Invalid session")
            return

        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        samples = self.sess.run([self.sampler], feed_dict={self.z: sample_z})
        samples = np.reshape(samples, (-1, 28 * 28))
        sample_inputs = self.data_X[0:50]
        data = np.reshape(sample_inputs, (-1, 28 * 28))
        # from IPython import embed; embed()
        im_size = 28
        n_clusters = 4
        n_fig_unit = 2
        out_path = os.path.abspath('./out_clustering/out_gen')
        inp_path = os.path.abspath('./out_clustering/inp_training')
        reconstruct_image(samples, out_path, True, im_size, n_clusters,
                          n_fig_unit, 10)
        reconstruct_image(data, inp_path, True, im_size, n_clusters,
                          n_fig_unit, 10)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(
                    h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(
                    s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(
                    s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(
                    s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(
                    s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                # h4_flat = tf.reshape(h4, [-1, s_h * s_w])
                # pkeep = 0.5
                # self.G_W1 = tf.Variable(self.initializer(shape=[s_h * s_w, 406]),
                #                         name='g_h5')
                # self.G_b1 = tf.Variable(tf.zeros(shape=[406]), name='g_b5')
                #
                # self.G_W2 = tf.Variable(
                #     self.initializer([406, 28 * 28]), name='g_h6')
                # self.G_b2 = tf.Variable(tf.zeros(shape=[28 * 28]), name='g_b6')
                # G_h1 = tf.nn.relu(tf.matmul(h4_flat, self.G_W1) + self.G_b1)
                # # G_h1 = tf.nn.dropout(G_h1, pkeep)
                # G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
                # # G_h2 = tf.nn.dropout(G_h2, pkeep)
                # out = tf.reshape(G_h2, [-1, s_h, s_w, self.c_dim])

                return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(
                    s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(
                    s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(
                    s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(
                    s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8,
                                   s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4,
                                   s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2,
                                   s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h,
                                   s_w, self.c_dim], name='g_h4')

                # h4_flat = tf.reshape(h4, [-1, s_h * s_w])
                # pkeep = 0.5
                # self.G_W1 = tf.Variable(self.initializer(shape=[s_h * s_w, 406]),
                #                         name='g_h5')
                # self.G_b1 = tf.Variable(tf.zeros(shape=[406]), name='g_b5')
                #
                # self.G_W2 = tf.Variable(
                #     self.initializer([406, 28 * 28]), name='g_h6')
                # self.G_b2 = tf.Variable(tf.zeros(shape=[28 * 28]), name='g_b6')
                # G_h1 = tf.nn.relu(tf.matmul(h4_flat, self.G_W1) + self.G_b1)
                # # G_h1 = tf.nn.dropout(G_h1, pkeep)
                # G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
                # # G_h2 = tf.nn.dropout(G_h2, pkeep)
                # out = tf.reshape(G_h2, [-1, s_h, s_w, self.c_dim])

                return tf.nn.tanh(h4)

    def load_mnist(self):
        filename = '../experiments/dataset/sbm_4p10.npy'
        data = np.load(filename)
        data = np.reshape(data, (-1, 28, 28, 1))
        return data

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
