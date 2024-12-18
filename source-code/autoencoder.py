import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from layers import ZINBLoss, MeanAct, DispAct
from torch.nn.parameter import Parameter


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,
                 n_dec_1, n_dec_2,
                 n_input, n_z,
                 denoise, sigma):
        super(AE, self).__init__()
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.n_input = n_input
        self.n_z = n_z
        self.denoise = denoise
        self.sigma = sigma

        self.enc_1 = Linear(self.n_input, self.n_enc_1, bias=True)
        self.enc_2 = Linear(self.n_enc_1, self.n_enc_2, bias=True)
        # self.enc_3 = Linear(n_enc_2, n_enc_3, bias=True)
        self._enc_mu = Linear(self.n_enc_2, self.n_z, bias=True)
        # self.z_layer = Linear(self.n_enc_2, self.n_z, bias=True)

        self.dec_1 = Linear(self.n_z, self.n_dec_1, bias=True)
        self.dec_2 = Linear(self.n_dec_1, self.n_dec_2, bias=True)
        # self.dec_3 = Linear(n_dec_2, n_dec_3)
        self._dec_mean = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), nn.Sigmoid())
        self._dec_beta = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), DispAct())
        self._dec_gamma = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), DispAct())
        self._dec_alpha = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), DispAct())
        self.x_bar_layer = Linear(self.n_dec_2, self.n_input, bias=True)

    def dropoutlayer(self, x, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1:
            return torch.zeros_like(x)
        if dropout == 0:
            return x
        mask = (torch.rand(x.shape) > dropout).float()
        return mask * x / (1.0 - dropout)

    def Encoder(self, x):
        if not self.denoise:
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            # enc_h3 = F.relu(self.enc_3(enc_h2))
            z = self._enc_mu(enc_h2)
        else:
            x = x + torch.randn_like(x) * self.sigma
            enc_h1 = F.relu(self.enc_1(x))
            # enc_h1 = self.dropoutlayer(enc_h1, 0.05)
            enc_h2 = F.relu(self.enc_2(enc_h1))
            # enc_h2 = self.dropoutlayer(enc_h2, 0.05)
            # enc_h3 = F.relu(self.enc_3(enc_h2))
            z = self._enc_mu(enc_h2)
        return z  # enc_h1, enc_h2, enc_h1_hat, enc_h2_hat

    def Decoder(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))
        mean = self._dec_mean(dec_h2)
        disp = self._dec_disp(dec_h2)
        pi = self._dec_pi(dec_h2)
        beta = self._dec_beta(dec_h2)
        gamma = self._dec_gamma(dec_h2)
        alpha = self._dec_alpha(dec_h2)
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, mean, disp, pi, beta, gamma, alpha

    def forward(self, x):
        z = self.Encoder(x)
        x_bar, mean, disp, pi, beta, gamma, alpha = self.Decoder(z)
        return z, x_bar, mean, disp, pi, beta, gamma, alpha


class VAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,
                 n_dec_1, n_dec_2,
                 n_input, n_z,
                 denoise):
        super(VAE, self).__init__()
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.n_input = n_input
        self.n_z = n_z
        self.denoise = denoise

        self.enc_1 = Linear(self.n_input, self.n_enc_1, bias=True)
        self.enc_2 = Linear(self.n_enc_1, self.n_enc_2, bias=True)
        self.enc_mu = Linear(self.n_enc_2, self.n_z, bias=True)
        self.enc_sigma = Linear(self.n_enc_2, self.n_z, bias=True)

        self.dec_1 = Linear(self.n_z, self.n_dec_1, bias=True)
        self.dec_2 = Linear(self.n_dec_1, self.n_dec_2, bias=True)
        self.x_bar_layer = nn.Sequential(nn.Linear(self.n_dec_2, self.n_input), nn.Sigmoid())

    def dropoutlayer(self, x, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1:
            return torch.zeros_like(x)
        if dropout == 0:
            return x
        mask = (torch.rand(x.shape) > dropout).float()
        return mask * x / (1.0 - dropout)

    def Encoder(self, x):
        if not self.denoise:
            enc_h1 = F.relu(self.enc_1(x))
            enc_h2 = F.relu(self.enc_2(enc_h1))
            mu = self.enc_mu(enc_h2)
            sigma = self.enc_sigma(enc_h2)
        else:
            '''
            x = x + torch.randn_like(x) * self.sigma
            '''
            enc_h1 = F.relu(self.enc_1(x))
            enc_h1 = self.dropoutlayer(enc_h1, 0.2)
            enc_h2 = F.relu(self.enc_2(enc_h1))
            enc_h2 = self.dropoutlayer(enc_h2, 0.2)
            mu = self.enc_mu(enc_h2)
            sigma = self.enc_sigma(enc_h2)
        return mu, sigma

    def Decoder(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar

    def forward(self, x):
        mu, sigma = self.Encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        x_bar = self.Decoder(z)
        return z, mu, sigma, x_bar
