# coding: utf-8

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from make_nonliner_dataset import get_xor_dataset
from plot_hyperplane import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

X, y = get_xor_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

svm = SVC(kernel='rbf', gamma=0.10, C=10.0, random_state=0)
svm.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=svm)

plt.legend(loc='upper left')
plt.show()
