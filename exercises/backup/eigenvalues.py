import numpy as np
from scipy import linalg

A = np.array([[0,3,0],[0,5,0],[0,0,2]])

#B = np.linalg.eig(A)
B = linalg.eig(A)
print(B)