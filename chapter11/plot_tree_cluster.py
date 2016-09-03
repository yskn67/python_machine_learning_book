# coding: utf-8

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from get_tree_data import get_labels, get_tree_data, get_dist_data
import pandas as pd
import matplotlib.pyplot as plt


labels = get_labels()
df = get_tree_data()
dist = get_dist_data(df)
row_clusters = linkage(dist, method='complete')

pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])])

row_dendr = dendrogram(row_clusters,
                       labels=labels)

plt.ylabel('Euclidean distance')
plt.show()

fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.090, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='right')
df_rowclust = df.ix[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)

fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

from sklearn.cluster import AgglomerativeClustering


ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(df)
print('Cluster labels: %s' % labels)
