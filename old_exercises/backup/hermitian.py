import numpy as np

A = np.matrix("1+2j,3+4j;3+6j,4-8j")

B = A.T
C = A.H

print(A)
print(B)
print(C)
