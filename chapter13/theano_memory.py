# coding: utf-8

import theano
from theano import tensor as T
import numpy as np


x = T.dmatrix(name='x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

net_input = theano.function(inputs=[x], updates=update, outputs=z)

data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z%d:' % i, net_input(data))
