# coding: utf-8

import sys
import os
sys.path.append(os.pardir)

from load_mnist import load_mnist
from neural_net_mlp import MLPGradientCheck


X_train, y_train = load_mnist('../mnist', kind='train')
X_test, y_test = load_mnist('../mnist', kind='t10k')

nn_check = MLPGradientCheck(n_output=10,
                            n_features=X_train.shape[1],
                            n_hidden=10,
                            l2=0.0,
                            l1=0.0,
                            epochs=10,
                            eta=0.001,
                            alpha=0.0,
                            decrease_const=0.0,
                            minibatches=1,
                            random_state=1)

nn_check.fit(X_train[:5], y_train[:5], print_progress=False)
