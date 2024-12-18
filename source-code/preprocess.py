import csv
import h5py
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

eps = 2.2204e-16


def filter_genes_zero(data):
    n_samples, n_genes = data.shape[0], data.shape[1]
    gene_nonzero = [False if (np.unique(data[:, col]) == [0]).all() else True for col in range(n_genes)]
    return data[:, gene_nonzero]


def matrixNormalize(x):
    # normalize cells
    cells_norm2 = np.linalg.norm(x, axis=0)
    return x / cells_norm2, cells_norm2


def errorLinSys(P, Z):
    R, N = Z.shape[0], Z.shape[1]
    if R > N:
        E = P[:, N + 1:].dot(Z[N + 1:])
        Y = P[:, :N]
        Y0 = Y - E
        C = Z[:N]
    else:
        Y = P
        Y0 = P
        C = Z
    Yn, norm = matrixNormalize(Y0)
    norm = norm.reshape((1, -1))
    M = np.tile(norm, (Y.shape[0], 1))
    S = Yn - Y.dot(C) / M
    err = np.sqrt(np.max(np.sum(S ** 2, axis=0)))
    return err


def computeLambda_mat(Y, P=None):
    if P is None:
        P = Y.copy()
    n_samples = Y.shape[1]
    T = P.T.dot(Y)
    T[:n_samples] = T[:n_samples] - np.diag(T[:n_samples].diagonal())
    lamda = np.min(np.max(T, 0))
    return lamda


def errorCoef(Z, C):
    return np.abs(Z - C).max()


def admmLasso_mat_func(Y, affine=False, alpha=800, thr=2e-4, maxIter=200):
    if isinstance(alpha, int) or isinstance(thr, float):
        alpha = [alpha]
    if isinstance(thr, float) or isinstance(alpha, int):
        thr = [thr]

    alpha1, alpha2 = alpha[0], alpha[-1]
    thr1, thr2 = thr[0], thr[-1]
    n_samples = Y.shape[1]

    # setting penalty
    mu1 = alpha1 * 1 / computeLambda_mat(Y)
    mu2 = alpha2

    if not affine:
        # initialization
        A = np.linalg.inv(mu1 * (Y.T.dot(Y)) + mu2 * np.eye(n_samples))
        C1 = np.zeros(shape=(n_samples, n_samples))
        lamda2 = np.zeros(shape=(n_samples, n_samples))
        err1, err2 = [10 * thr1], [10 * thr2]
        i = 0
        while err1[i] > thr1 and i < maxIter:
            # updating z
            Z = A.dot(mu1 * (Y.T.dot(Y)) + mu2 * (C1 - lamda2 / mu2))
            Z = Z - np.diag(Z.diagonal())

            # updating c
            tmp = np.abs(Z + lamda2 / mu2) - 1 / mu2 * np.ones(shape=(n_samples, n_samples))
            C2 = np.where(tmp >= 0, tmp, np.zeros((tmp.shape))) * np.sign(Z + lamda2 / mu2)
            C2 = C2 - np.diag(C2.diagonal())

            # updating lagrnge multipliers
            lamda2 += mu2 * (Z - C2)

            # computing errors
            err1.append(errorCoef(Z, C2))
            err2.append(errorLinSys(Y, Z))

            C1 = C2.copy()
            i += 1
    else:
        # initialization
        A = np.linalg.inv(mu1 * (Y.T.dot(Y)) + mu2 * np.eye(n_samples) + mu2 * np.ones((n_samples, n_samples)))
        C1 = np.zeros(shape=(n_samples, n_samples))
        lamda2 = np.zeros(shape=(n_samples, n_samples))
        lamda3 = np.zeros(shape=(1, n_samples))
        err1, err2, err3 = [10 * thr1], [10 * thr2], [10 * thr1]
        i = 1
        while (err1[i] > thr1 or err3[i] > thr1) and i < maxIter:
            # updating z
            Z = A.dot(mu1 * (Y.T.dot(Y)) + mu2 * (C1 - lamda2 / mu2) + mu2 * np.ones((n_samples, 1)).dot(
                np.ones((1, n_samples)) - lamda3 / mu2))
            Z = Z - np.diag(Z.diagonal())

            # c
            tmp = np.abs(Z + lamda2 / mu2) - 1 / mu2 * np.ones((n_samples, n_samples))
            C2 = np.max(tmp >= 0, tmp, np.zeros(tmp.shape)) * np.sign(Z + lamda2 / mu2)
            C2 = C2 - np.diag(C2.diagonal())

            # lagrange multipliers
            lamda2 += mu2 * (Z - C2)
            lamda3 += mu2 * (np.ones((1, n_samples)).dot(Z) - np.ones((1, n_samples)))

            # errors
            err1.append(errorCoef(Z, C2))
            err2.append(errorLinSys(Y, Z))
            err3.append(errorCoef(np.ones((1, n_samples)).dot(Z), np.ones((1, n_samples))))

            #
            C1 = C2.copy()
            i += 1
        print('err1: {:.4f}, err2:{:.4f}, err3:{:.4f}, iter:{}'.format(err1[-1], err2[-1], err3[-1], i))
    return C2


def eps2C(x, c=10000):
    return np.where(x >= 1e-12, x, c * np.ones(shape=x.shape))


def LaplacianScore(x, w):
    # x in (samples, features)
    n_samples, n_feat = x.shape[0], x.shape[1]

    if w.shape[0] != n_samples:
        raise Exception("W.shape not match X.shape")

    D = np.diag(np.sum(w, axis=1))  # (n_samples,)
    D2 = np.sum(w, axis=1)  # (n_samples,)
    L = w

    tmp1 = (D2.T).dot(x)
    DPrime = np.sum((x.T.dot(D)).T * x, axis=0) - tmp1 * tmp1 / np.sum(D2)
    LPrime = np.sum((x.T.dot(L)).T * x, axis=0) - tmp1 * tmp1 / np.sum(D2)

    DPrime = eps2C(DPrime, c=10000)
    a1 = np.sum(D)
    a2 = np.sum((x.T.dot(D)).T * x, axis=0)
    a3 = tmp1 * tmp1 / np.sum(D)
    a4 = (x.T.dot(D)).T * x
    a7 = ((x.T).dot(D)).T * x
    a5 = tmp1 * tmp1
    a6 = x.T.dot(D)
    a9 = np.dot(x.T, D)

    Y = LPrime / DPrime
    # Y = Y.T#lzl edit
    return Y


def cosine(data):
    cosine = 1 - pairwise_distances(data, metric='cosine')
    return np.where((cosine <= 1) & (cosine >= 0), cosine, np.zeros(cosine.shape))


def euclidean(data):
    dist = 1 - pairwise_distances(data, metric='euclidean')
    # dist transform sim
    # sim = np.exp(-dist/dist.max())
    return np.where((dist <= 1) & (dist >= 0), dist, np.zeros(dist.shape))


def pearson(data):
    df = pd.DataFrame(data.T)
    pear_ = df.corr(method='pearson')  # 返回的是列与列之间的相关系数，所以提前使用转置，将细胞*基因的矩阵转为基因*细胞，列表示细胞
    return np.where(pear_ >= 0, pear_, np.zeros(shape=pear_.shape))


def spearman(data):
    df = pd.DataFrame(data.T)
    spear_ = df.corr(method='spearman')
    return np.where(spear_ >= 0, spear_, np.zeros(shape=spear_.shape))


def all_similarities(data):
    return pearson(data), spearman(data), cosine(data), euclidean(data)


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
        select_index = np.argmin(gene_var)  # 取最小值对应的索引值
        gene_inter.append(sort_ind[:select_index])  # 取出了前t个基因
        gene_inter_num.append(select_index)  # 取出的基因数量

    gene_slect_ssc_pear = np.intersect1d(gene_inter[0], gene_inter[1])  # 取两个数组中相同的值
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
    # print("m,n:", (m, n))
    if m > 5000:
        edge_num = 100
    else:
        edge_num = int(np.round(m * 0.1))

    index1 = np.argsort(pear_sim, axis=1)  # 按列从小到大排序
    pear_index = index1[:, -edge_num:]  # 取后num个索引，相当于取出相似度大的细胞

    index2 = np.argsort(spear_sim, axis=1)
    spear_index = index2[:, -edge_num:]

    index3 = np.argsort(cos_sim, axis=1)
    cos_index = index3[:, -edge_num:]
    selected_edge_combine = {0: pear_index, 1: spear_index, 2: cos_index}

    elected_edge = []
    for i in range(m):
        tmp = np.union1d(pear_index[i, :], spear_index[i, :])  # 取并集
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
                            RA = RA + 1 / neighbor_num  # 传统AA
                            WRA = WRA + (data[i, z] + data[select_edge[i][j], z]) / neighbor_num  # 加权的AA
                RA_score[i, select_edge[i][j]] = RA
                WRA_score[i, select_edge[i][j]] = WRA
    WRA_value = WRA_score + WRA_score.T
    test_data0 = data + WRA_value
    test_data = test_data0 - np.diag(test_data0.diagonal())
    return test_data


def spectralclustering(data, n_clusters):
    N = data.shape[0]
    maxiter = 1000  # max iteration times
    replic = 100  # number of time kmeans will be run with diff centroids

    DN = np.diag(1 / np.sqrt(np.sum(data, axis=0) + eps))
    lapN = np.eye(N) - DN.dot(data).dot(DN)
    U, A, V = np.linalg.svd(lapN)
    V = V.T
    kerN = V[:, N - n_clusters:N]
    normN = np.sum(kerN ** 2, 1) ** 0.5
    kerNS = (kerN.T / (normN + eps)).T
    # kmeans
    clf = KMeans(n_clusters=n_clusters, max_iter=maxiter, n_init=replic)
    return clf.fit_predict(kerNS)


data = pd.read_csv('Darmanis.csv', header=None)
# data = np.transpose(data)
data = np.array(data)
label = pd.read_csv('Darmanis_label.csv', header=None)
label = np.array(label)
label = label.ravel()

'''
data_mat = h5py.File('CITE_CBMC_counts_top2000.h5')
data = np.array(data_mat['X'])
# data = data.T
label = np.array(data_mat['Y'])
data_mat.close()
label = label.ravel()
'''
'''
data = pd.read_table('Test_9_Yan.txt', header=None, sep='\t')
data = np.array(data)
label = pd.read_table('Test_9_Yan_label.txt', header=None, sep='\t')
label = np.array(label)
label = label.ravel()
'''

# 计算对相似性
n_clusters = len(set(label))
data = filter_genes_zero(data)  # row is a cell
pairwise_data0 = data
pear_sim0, spear_sim0, cos_sim0, _ = all_similarities(pairwise_data0)

# 计算基因分数
pear_score = LaplacianScore(pairwise_data0, pear_sim0)
spear_score = LaplacianScore(pairwise_data0, spear_sim0)
cos_score = LaplacianScore(pairwise_data0, cos_sim0)

# 计算稀疏相似性
ssc_data0, _ = matrixNormalize(data.T)  # (features, samples)
CMat3 = admmLasso_mat_func(ssc_data0, False, 10)
C0 = np.abs(CMat3) + np.abs(CMat3.T)
ssc_score = LaplacianScore(ssc_data0.T, C0)

# 调和均值计算
select_Laplacianscore = (4 * pear_score * spear_score * cos_score * ssc_score) / (
        pear_score + spear_score + cos_score + ssc_score)

indexed_array = list(enumerate(select_Laplacianscore))
sorted_indexed_array = sorted(indexed_array, key=lambda x: x[1], reverse=True)
sorted_indices = [x[0] for x in sorted_indexed_array]

gene_select = sorted_indices[:2000]
data_select = data[:, gene_select]
print("data_select.shape:", data_select.shape)

with open('Pollen_select1.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in data_select:
        writer.writerow(row)
    print("finish!")
