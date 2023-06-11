from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrroms_ix, population_ix, households_ix = 3, 4, 5 ,6


array1 = [0,1,2]
array2 = [3,4,5]

a = np.c_(np.array([0,1,2]), np.array([3,4,5]))
print(a)
