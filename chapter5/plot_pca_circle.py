# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA


X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_pca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_pca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
