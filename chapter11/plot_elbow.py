# coding: utf-8

from sklearn.cluster import KMeans
from get_data import get_data
import matplotlib.pyplot as plt


X, y = get_data()
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=100,
                random_state=0)
    y_km = km.fit_predict(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
