# coding: utf-8

import theano
from theano import tensor as T
import numpy as np


x = T.dmatrix(name='x')
x_sum = T.sum(x, axis=0)

calc_sum = theano.function(inputs=[x], outputs=x_sum)

ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))
