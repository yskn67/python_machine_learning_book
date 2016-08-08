# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import plot_iris
from ml_class import AdalineSGD
from plot_hyperplane import plot_decision_regions

X, y = plot_iris.get_x_y()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(eta=0.01, n_iter=15, random_state=1)

ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.show()
