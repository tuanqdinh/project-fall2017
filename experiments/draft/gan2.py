import torch.optim as optim
from torch.autograd import Variable
import pandas as pd

from models.model1 import *
from helper import *


def get_data():
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    mb_size = 64
    Z_dim = 100
    X_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    h_dim = 128

def evaluate():


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    return dis_loss, gen_loss




if __name__ == '__main__':
    
    helper = Helper()

    G = Generator(input_size=helper.g_input_size, 
        hidden_size=helper.g_hidden_size, output_size=helper.g_output_size)
    D = Discriminator(input_size=1, 
        hidden_size=helper.d_hidden_size, output_size=helper.d_output_size)

    criterion = nn.BCELoss()  

    G_solver = optim.Adam(G_params, lr=1e-3)
    D_solver = optim.Adam(D_params, lr=1e-3)

    for it in range(1000):

        # Sample data
        z = Variable(torch.randn(mb_size, Z_dim))
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        D.zero_grad()
        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient
        G.zero_grad()

        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = nn.binary_cross_entropy(D_fake, ones_label)

        G_loss.backward()
        G_solver.step()

        # Housekeeping - reset gradient
        reset_grad()
