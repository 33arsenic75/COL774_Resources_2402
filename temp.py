import numpy as np
A = np.array([[3, 5], [1, 2]])
# A_inv = np.array([[2, -5], [-1, 3]])
A_inv = np.array([[2, 22], [26, 3]])


v = [ np.array([13 , 20]),
      np.array([12 , 0]),
      np.array([12 , 9]),
      np.array([14 , 5]),
      np.array([1  , 18]),
      np.array([0  , 1]),
      np.array([12 , 7]),
      np.array([5  , 2]),
      np.array([18 , 1]) ]

for i in range(len(v)):
    print(np.dot(A_inv, np.dot(A, v[i]))%27)