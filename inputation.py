import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [27, 'f'],
    [np.nan, np.nan]
]

trans = [
    ['age', SimpleImputer(strategy='median'), [0]],
    ['sex', SimpleImputer(strategy='constant', fill_value='N.D.'), [1]],
]

ct = ColumnTransformer(trans)
X = ct.fit_transform(X)

print(X)
