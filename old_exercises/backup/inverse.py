import numpy as np
import scipy.linalg

A = np.matrix("4,7;2,6")
B = scipy.linalg.inv(A)
C = scipy.linalg.det(A)


print(A)
print(B)
print(C)

