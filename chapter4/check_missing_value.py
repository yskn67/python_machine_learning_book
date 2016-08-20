# coding: utf-8

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
# print data including NaN value
print(df, "\n")
# count NaN value
print(df.isnull().sum(), "\n")
# remove row including NaN value
print(df.dropna(), "\n")
# remove column including NaN value
print(df.dropna(axis=1), "\n")
# complement NaN value
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data, "\n")
