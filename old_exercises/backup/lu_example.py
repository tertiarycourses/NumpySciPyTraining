import numpy as np
from scipy import linalg

A = np. matrix([[ 12, -51,   4],
               [  6, 167, -68],
               [ -4,  24, -41]])

print(A)

P,L,U = linalg.lu(A)
print(np.dot(P,np.dot(L,U)))
