# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../housing.data')
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()
