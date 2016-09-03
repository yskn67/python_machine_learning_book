# coding: utf-8

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_labels():
    return ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']


def get_tree_data():
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = get_labels()
    X = np.random.random_sample([5, 3]) * 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    return df


def get_dist_data(df):
    return pdist(df, metric='euclidean')


def get_dist_matrix(df):
    labels = get_labels()
    dist = get_dist_data(df)
    row_dist = pd.DataFrame(squareform(dist),
                            columns=labels, index=labels)
    return row_dist


if __name__ == '__main__':
    df = get_tree_data()
    print(df)
    dist = get_dist_data(df)
    print(dist)
    row_dist = get_dist_data(df)
    print(row_dist)
