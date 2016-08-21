# coding: utf-8

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh


def rbf_kernel_pca(X, gamma, n_components):

    # M*N次元のデータセットでペアごとの平方ユークリッド距離を計算
    sq_dists = pdist(X, 'sqeuclidean')
    # ペアごとの距離を正方行列に変換
    mat_sq_dists = squareform(sq_dists)
    # 大正カーネル行列を計算
    K = exp(-gamma * mat_sq_dists)
    # カーネル行列の中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 中心化されたカーネル行列から固有値を取得
    # numpy.eighはそれらをソート順に返す
    eigvals, eigvecs = eigh(K)
    # 上位k個の固有ベクトル（射影されたサンプル）を収集
    alphas = np.column_stack((eigvecs[:, -i]
                              for i in range(1, n_components + 1)))
    # 対応する固有値を収集
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas
