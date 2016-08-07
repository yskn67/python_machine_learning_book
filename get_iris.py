# coding: utf-8

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.to_csv('iris.data', header=False, index=False)
