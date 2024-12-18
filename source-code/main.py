import matplotlib.cm
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from layers import ZINBLoss, MeanAct, DispAct, TPGGLoss, colwise, SliceLayer
from evaluation import eva
from autoencoder import AE, VAE
from preprocess import normalize
import scanpy as sc
from utils import thrC, post_proC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import savefig
import h5py
import csv

class Deep_Sparse_Subspace_Clustering(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, denoise, sigma, pre_lr, alt_lr,
                 adata, pre_epoches, alt_epoches,
                 lambda_1, lambda_2):
        super(Deep_Sparse_Subspace_Clustering, self).__init__()
        self.n_enc_1 = n_enc_1  # 编码器第一层
        self.n_enc_2 = n_enc_2  # 编码器第二层
        self.n_dec_1 = n_dec_1  # 解码器第一层
        self.n_dec_2 = n_dec_2  # 解码器第二层
        self.n_input = n_input  # 输入维度
        self.n_z = n_z  # 隐藏层层数
        self.denoise = denoise  # 是否加入噪声
        self.sigma = sigma
        self.pre_lr = pre_lr  # 预训练的学习率
        self.alt_lr = alt_lr
        self.adata = adata  # 输入的数据
        self.pre_epoches = pre_epoches  # 预训练的次数
        self.alt_epoches = alt_epoches
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.model = AE(n_enc_1=self.n_enc_1, n_enc_2=self.n_enc_2,
                        n_dec_1=self.n_dec_1, n_dec_2=self.n_dec_2,
                        n_input=self.n_input, n_z=self.n_z,
                        denoise=self.denoise, sigma=self.sigma)
        self.zinb_loss = ZINBLoss()
        self.TPGG_loss = TPGGLoss()
        self.sizelayer = torch.tensor([1.0])

        weights = self._initialize_weights()
        self.Coef = weights['Coef']

    def _initialize_weights(self):  # C(自表达矩阵)
        all_weights = dict()  # 创建一个字典
        all_weights['Coef'] = Parameter(
            1.0e-4 * (torch.ones(size=(len(self.adata.X), len(self.adata.X))) - torch.eye(len(self.adata.X))))
        return all_weights

    def pre_train(self):
        self.model.train()
        log_interval = 1
        optimizer = Adam(self.parameters(), lr=self.pre_lr)
        for epoch in range(1, self.pre_epoches + 1):
            x_tensor = Variable(torch.Tensor(self.adata.X))
            # print("tensor:",x_tensor.shape)
            x_raw_tensor = Variable(torch.Tensor(self.adata.raw.X))
            sf_tensor = Variable(torch.Tensor(self.adata.obs.size_factors))
            z, x_bar, mean, disp, pi, beta, gamma, alpha = self.model(x_tensor)
            # print("z:",z.shape)
            output = colwise([alpha, self.sizelayer])
            output = SliceLayer(0)([output, beta, gamma, pi])
            z_c = torch.matmul(self.Coef, z)  # Xhat=XC 自表达矩阵
            loss_reconst = 1 / 2 * torch.sum(torch.pow((x_tensor - x_bar), 2))
            loss_reg = torch.sum(torch.pow(self.Coef, 2))
            loss_selfexpress = 1 / 2 * torch.sum(torch.pow((z - z_c), 2))
            loss_zinb = self.zinb_loss(x=x_raw_tensor, mean=mean, disp=disp, pi=pi, scale_factor=sf_tensor)
            loss_tpgg = self.TPGG_loss(x=x_raw_tensor, x_hat=output, pi_hat=pi, beta_hat=beta, gamma_hat=gamma)
            loss = (
                           0.2 * loss_reconst + self.lambda_1 * loss_reg + self.lambda_2 * loss_selfexpress) ** 1 / 10 + loss_tpgg
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            if epoch % log_interval == 0:
                print('Train Epoch: {} ''\tLoss: {:.6f}'.format(epoch, loss.item()))
            if epoch == self.pre_epoches:
                print('Pre-training completed')
        return self.Coef.detach().numpy()


x_hat = pd.read_csv('Haber_select1.csv', header=None)
x_hat = np.array(x_hat)
y = pd.read_csv('Haber_label.csv', header=None)
y = np.array(y)
y = y.ravel()

adata_hat = sc.AnnData(x_hat)
print("adata:", adata_hat.shape)
sc.pp.filter_genes(adata_hat, min_counts=0)
sc.pp.filter_cells(adata_hat, min_counts=0)
sc.pp.log1p(adata_hat)
# sc.pp.normalize_per_cell(adata_hat)
adata_hat.raw = adata_hat.copy()
adata_hat.obs['size_factors'] = adata_hat.obs.n_counts / np.median(adata_hat.obs.n_counts)
adata_hat.obs['Group'] = y

x_sd_hat = adata_hat.X.std(0)
x_sd_median_hat = np.median(x_sd_hat)
sd = 2.5
inputsize = adata_hat.X.shape[1]

net_hat = Deep_Sparse_Subspace_Clustering(n_enc_1=256, n_enc_2=32, n_dec_1=32, n_dec_2=256, n_input=inputsize,
                                          n_z=10, denoise=False, sigma=2.0, pre_lr=0.001, alt_lr=0.001,
                                          adata=adata_hat, pre_epoches=300, alt_epoches=100, lambda_1=1.0, lambda_2=0.5)

Coef_1_hat = net_hat.pre_train()
Coef_1_hat_test = np.matmul(Coef_1_hat, Coef_1_hat) + Coef_1_hat

pred_label_hat, _ = post_proC(Coef_1_hat_test, 8)
pred_label_hat = pred_label_hat.astype(np.int64)
y = y.astype(np.int64)
eva(y, pred_label_hat)
