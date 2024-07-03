import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def colwise(l):
    m1 = l[0]
    m2 = torch.reshape(l[1], (-1, 1))
    return m1 * m2


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class SliceLayer(nn.Module):
    def __init__(self, index):
        super(SliceLayer, self).__init__()
        self.index = index

    def forward(self, x):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class TPGGLoss(nn.Module):
    def __init__(self):
        super(TPGGLoss, self).__init__()

    def forward(self, x, x_hat, pi_hat, beta_hat, gamma_hat):
        eps = 1e-6
        scale_factor = 1.0
        x_hat = x_hat * scale_factor
        beta_hat = torch.minimum(beta_hat, torch.tensor(1e6))
        gamma_hat = torch.minimum(gamma_hat, torch.tensor(1e6))
        p1 = torch.log(x_hat + eps) + torch.lgamma(beta_hat + eps) - torch.log(gamma_hat + eps) - (
                beta_hat * gamma_hat - 1) * (torch.log(x + eps) - torch.log(x_hat + eps))
        p2 = (x / x_hat + eps) ** gamma_hat
        p = p1 + p2
        p = _nan2inf(p)
        p = torch.mean(p)
        gg_case = p - torch.log(1.0 - pi_hat + eps)
        zero_case = -torch.log(pi_hat + eps)
        result = torch.where(torch.less(x, 1e-8), zero_case, gg_case)
        result = torch.mean(result)
        result = _nan2inf(result)
        return result


'''
class TPGGLoss(nn.Module):
    def __init__(self):
        super(TPGGLoss, self).__init__()

    def forward(self, x, x_hat, pi_hat, beta_hat, gamma_hat):
        eps = 1e-6
        scale_factor = 1.0
        x_hat = x_hat * scale_factor
        # beta_hat = torch.minimum(beta_hat, 1e6)
        # gamma_hat = torch.minimum(gamma_hat, 1e6)
        p1 = torch.log(x_hat + eps) + torch.lgamma(beta_hat + eps) - torch.log(gamma_hat + eps) - (
                beta_hat * gamma_hat - 1) * (torch.log(x + eps) - torch.log(x_hat + eps))
        p2 = (x / x_hat + eps) ** gamma_hat
        p = p1 + p2
        p = torch.mean(p)
        gg_case = p - torch.log(1.0 - pi_hat + eps)
        zero_case = -torch.log(pi_hat + eps)
        result = torch.where(torch.less(x, 1e-8), zero_case, gg_case)
        result = torch.mean(result)
        return result
'''


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
