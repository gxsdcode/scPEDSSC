import numpy as np
from sklearn.cluster import KMeans

eps = 2.2204e-16


def l_gene_select_combine(ssc_score, pear_score, spear_score, cos_score):
    score_set = [ssc_score, pear_score, spear_score, cos_score]
    gene_inter, gene_inter_num = [], []
    for i in range(4):
        # ascend sort
        score, sort_ind = np.sort(score_set[i]), np.argsort(score_set[i])
        # descend sort
        score, sort_ind = score[::-1], sort_ind[::-1]
        gene_num = len(score_set[i])
        thresh1 = int(np.round(0.1 * gene_num))
        thresh2 = int(np.round(0.5 * gene_num))

        gene_var = np.zeros((thresh2 + 1,))
        for j in np.arange(thresh1, thresh2 + 1):
            score1 = score[:j]
            score2 = score[j:]
            var1 = score1.var()
            var2 = score2.var()
            gene_var[j] = var1 + var2
        gene_var[:thresh1] = np.inf
        select_index = np.argmin(gene_var)
        gene_inter.append(sort_ind[:select_index])
        gene_inter_num.append(select_index)

    gene_slect_ssc_pear = np.intersect1d(gene_inter[0], gene_inter[1])
    gene_slect_ssc_spear = np.intersect1d(gene_inter[0], gene_inter[2])
    gene_slect_ssc_cos = np.intersect1d(gene_inter[0], gene_inter[3])
    # combine
    gene_slect_combine = {0: gene_slect_ssc_pear, 1: gene_slect_ssc_spear, 2: gene_slect_ssc_cos,
                          3: np.intersect1d(gene_slect_ssc_pear, gene_inter[2]),
                          4: np.intersect1d(gene_slect_ssc_pear, gene_inter[3]),
                          5: np.intersect1d(gene_slect_ssc_spear, gene_inter[3])}
    # combine

    gene_select = np.intersect1d(gene_slect_combine[3], gene_inter[3])
    return gene_select, gene_slect_combine, gene_inter_num


def l_choose_edge_combine(pear_sim, spear_sim, cos_sim):
    m, n = pear_sim.shape[0], pear_sim.shape[1]
    # print(m, n)
    if m > 5000:
        edge_num = 100
    else:
        edge_num = int(np.round(m * 0.1))

    index1 = np.argsort(pear_sim, axis=1)
    pear_index = index1[:, -edge_num:]

    index2 = np.argsort(spear_sim, axis=1)
    spear_index = index2[:, -edge_num:]

    index3 = np.argsort(cos_sim, axis=1)
    cos_index = index3[:, -edge_num:]
    # selected edge with different combination of similarities
    selected_edge_combine = {0: pear_index, 1: spear_index, 2: cos_index}

    elected_edge = []
    for i in range(m):
        tmp = np.union1d(pear_index[i, :], spear_index[i, :])
        elected_edge.append(np.sort(tmp))
    selected_edge_combine[3] = elected_edge

    elected_edge = []
    for i in range(m):
        tmp = np.union1d(pear_index[i, :], cos_index[i, :])
        elected_edge.append(np.sort(tmp))
    selected_edge_combine[4] = elected_edge

    elected_edge = []
    for i in range(m):
        tmp = np.union1d(spear_index[i, :], cos_index[i, :])
        elected_edge.append(np.sort(tmp))
    selected_edge_combine[5] = elected_edge
    return selected_edge_combine


def l_enhance(data, select_edge):
    data = np.abs(data)
    m, n = data.shape[0], data.shape[1]
    print('m,n:', m, n)
    test_data = data.copy()

    RA_score = np.zeros((m, n))
    WRA_score = np.zeros((m, n))

    for i in range(m):
        edge_len = len(select_edge[i])
        for j in range(edge_len):
            if data[i, select_edge[i][j]] == 0:
                RA, WRA = 0, 0
                for z in range(m):
                    if data[i, z] != 0 and data[select_edge[i][j], z] != 0:
                        neighbor_num = list(data[z] != 0).count(True)
                        if neighbor_num == 0:
                            print('a error happened')
                        else:
                            RA = RA + 1 / neighbor_num
                            WRA = WRA + (data[i, z] + data[select_edge[i][j], z]) / neighbor_num
                RA_score[i, select_edge[i][j]] = RA
                WRA_score[i, select_edge[i][j]] = WRA

    WRA_value = WRA_score + WRA_score.T
    test_data0 = data + WRA_value
    test_data = test_data0 - np.diag(test_data0.diagonal())
    return test_data


import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize as nor
from scipy.spatial import distance
import math
# from munkres import Munkres
# from means import kMeans, biKmeans
import pandas as pd
from sklearn.decomposition import PCA
import phate


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym


def spectralclustering(data, n_clusters):
    N = data.shape[0]
    maxiter = 1000  # max iteration times
    replic = 100  # number of time kmeans will be run with diff centroids

    DN = np.diag(1 / np.sqrt(np.sum(data, axis=0) + eps))
    lapN = np.eye(N) - DN.dot(data).dot(DN)
    U, A, V = np.linalg.svd(lapN)
    V = V.T
    kerN = V[:, N - n_clusters:N]
    normN = np.sum(kerN ** 2, 1) ** (0.5)
    kerNS = (kerN.T / (normN + eps)).T
    # kmeans
    clf = KMeans(n_clusters=n_clusters, max_iter=maxiter, n_init=replic)
    return clf.fit_predict(kerNS)


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))  # 奇异值分解，r为奇异值数量；返回的是左奇异值向量，奇异值（对角线的值）
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)  # 返回的矩阵为对角线值为S的元素，其余位置为0
    U = U.dot(S)  # 计算U*S
    U = nor(U, norm='l2', axis=1)  # l2范式处理矩阵
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    # K为聚类个数，初始化中心数在默认值10，affinity指定相似度矩阵的计算方式
    spectral.fit(L)  # 返回聚类对象本身
    grp = spectral.fit_predict(L) + 1  # 返回数据集预测标签
    return grp, L


def Conformal_mapping(X):
    a = np.zeros(X.shape[1])
    for i in range(0, X.shape[1]):
        a[i] = np.dot(X.T[i], np.ones(X.shape[0])) / X.shape[0]
    a = np.matrix(a)
    dists = distance.cdist(X, a, 'euclidean')
    R = np.max(dists)
    Xf = np.zeros_like(X)
    Xe = np.zeros(len(X))
    for i in range(0, len(X)):
        Xe[i] = R * ((np.dot(X[i] ** 2, np.ones(X.shape[1])) - math.pow(R, 2)) / (
                (np.dot(X[i] ** 2, np.ones(X.shape[1]))) + math.pow(R, 2)))
        Xf[i] = R * (2 * R / ((np.dot(X[i] ** 2, np.ones(X.shape[1]))) + math.pow(R, 2)) * X[i])

    X1 = np.column_stack((Xf, Xe))
    return X1


'''
def clust(data, pca_com):
    """
    input_path = data_path + ".csv"
    label_path = label_path + ".csv"
    X = pd.read_csv(input_path, header=None)
    X = X.drop(0)
    X = np.array(X)
    X = X.transpose()
    """
    X = data
    pca = PCA(n_components=pca_com)
    b = pca.fit_transform(X)
    # print(b.shape)
    phate_op = phate.PHATE(5)
    data_phate = phate_op.fit_transform(b)
    # print(data_phate.shape)
    # label = pd.read_csv(label_path, header=None)
    # scprep.plot.scatter2d(data_phate, c=label, ticks=None, label_prefix='PHATE', figsize=(5, 5),
    # cmap=sns.husl_palette(5))
    # y = np.array(label)
    # label = y.ravel()  # 将数组维度拉成一维
    # c = label.max()
    # print(c)
    centList, clusterAssment = biKmeans(data_phate, 14)
    # centList, clusterAssment = kMeans(data_phate, 13)
    julei = clusterAssment[:, 0]
    y = np.array(julei)
    julei = y.ravel()

    return julei
'''
