# coding: utf-8

import matplotlib.pyplot as plt
import plot_iris
from ml_class import Perceptron
from plot_hyperplane import plot_decision_regions

X, y = plot_iris.get_x_y()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plot_decision_regions(X, y, classifier=ppn)

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()
