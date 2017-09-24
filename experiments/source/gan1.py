import torch.optim as optim
from torch.autograd import Variable
import pandas as pd

from model import *
from helper import *


def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  
    # Gaussian


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  
    # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model
def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


def get_data():
    # Data params
    data_mean = 4
    data_mean = 4
    data_stddev = 1.25

    csvFile = pd.read_csv('../dataset/data_cog.csv')
    tData = csvFile.values
    data = tData[:100, :]
    testData = tData[100:, :]

    x = data[:, :-5]
    y = data[:, -4:-3] # Ab-142

    # torch can only train on Variable, so convert them to Variable
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x, y = Variable(x), Variable(y)

    return x,y


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    return dis_loss, gen_loss

def evaluate():
    testData = tData[100:, :]
    
    x = testData[:, :-5]
    y = testData[:, -4:-3] # Ab-142

    # torch can only train on Variable, so convert them to Variable
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x, y = Variable(x), Variable(y)
    prediction = G(x)



if __name__ == '__main__':
    
    helper = Helper()
    x, y = get_data()

    G = Generator(input_size=helper.g_input_size, 
        hidden_size=helper.g_hidden_size, output_size=helper.g_output_size)
    D = Discriminator(input_size=1, 
        hidden_size=helper.d_hidden_size, output_size=helper.d_output_size)

    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    # criterion = nn.MSELoss()
    d_optimizer = optim.Adam(D.parameters(), 
        lr=helper.d_learning_rate, betas=helper.optim_betas)
    g_optimizer = optim.Adam(G.parameters(), 
        lr=helper.g_learning_rate, betas=helper.optim_betas)

    for epoch in range(helper.num_epochs):
        for d_index in range(helper.d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = y
            d_real_decision = D(d_real_data)
            # d_real_data = Variable(d_sampler(d_input_size))
            # d_real_decision = D(preprocess(d_real_data))
            
            d_real_error = criterion(d_real_decision, Variable(torch.ones(100)))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = x
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            # d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(100)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     
            # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(helper.g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = x
            g_fake_data = G(gen_input)
            # dg_fake_decision = D(preprocess(g_fake_data.t()))
            dg_fake_decision = D(g_fake_data)
            g_error = criterion(dg_fake_decision, Variable(torch.ones(100)))  
            # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        if epoch % helper.print_interval == 0:
            print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                                extract(d_real_error)[0],
                                                                extract(d_fake_error)[0],
                                                                extract(g_error)[0],
                                                                stats(extract(d_real_data)),
                                                                stats(extract(d_fake_data))))


