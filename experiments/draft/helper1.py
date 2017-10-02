class Helper:
    def __init__(self):
        # Model params
        self.g_input_size = 27     # Random noise dimension coming into generator, per output vector
        self.g_hidden_size = 50   # Generator complexity
        self.g_output_size = 1    # size of generated output vector
        self.d_input_size = 1   # Minibatch size - cardinality of distributions
        self.d_hidden_size = 50   # Discriminator complexity
        self.d_output_size = 1    # Single dimension for 'real' vs. 'fake'
        self.minibatch_size = self.d_input_size

        self.d_learning_rate = 2e-4  # 2e-4
        self.g_learning_rate = 2e-4
        self.optim_betas = (0.9, 0.999)
        self.num_epochs = 300
        self.print_interval = 200
        self.d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
        self.g_steps = 1

    def get_distribution_sampler(mu, sigma):
        return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


    def get_generator_input_sampler():
        return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

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