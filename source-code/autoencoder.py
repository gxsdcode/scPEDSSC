import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from layers import ZINBLoss, MeanAct, DispAct
from torch.nn.parameter import Parameter


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,
                 n_dec_1, n_dec_2,
                 n_input, n_z):
        super(AE, self).__init__()
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_dec_1 = n_dec_1
        self.n_dec_2 = n_dec_2
        self.n_input = n_input
        self.n_z = n_z

        self.enc_1 = Linear(self.n_input, self.n_enc_1, bias=True)
        self.enc_2 = Linear(self.n_enc_1, self.n_enc_2, bias=True)
        self._enc_mu = Linear(self.n_enc_2, self.n_z, bias=True)

        self.dec_1 = Linear(self.n_z, self.n_dec_1, bias=True)
        self.dec_2 = Linear(self.n_dec_1, self.n_dec_2, bias=True)
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
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self._enc_mu(enc_h2)
        return z

    def Decoder(self, z):
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
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
