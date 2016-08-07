# coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')

    plt.legend(loc='upper left')
    plt.show()


def get_x_y():
    return (X, y)

df = pd.read_csv('../iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

if __name__ == '__main__':
    plot_data(X, y)
