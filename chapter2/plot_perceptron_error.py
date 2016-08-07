# coding: utf-8

import matplotlib.pyplot as plt
import plot_iris
from ml_class import Perceptron

X, y = plot_iris.get_x_y()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.show()
