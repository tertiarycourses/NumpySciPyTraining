import numpy as np
from scipy import linalg

A = np. matrix([[ 12, -51,   4],
               [  6, 167, -68],
               [ -4,  24, -41]])

print(A)
Q,R = linalg.qr(A)
print(np.dot(Q,R))
