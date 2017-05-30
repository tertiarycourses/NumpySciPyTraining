import numpy as np


A = np.matrix("1,2;3,4")
B = A.transpose()
C = A.T
print(A)
print(B)
print(C)

# A = np.matrix("1,2,3;4,5,6")
# B = np.matrix([[1,2,3],[4,5,6]]) 
# C = np.array([[1,2,3],[4,5,6]])
# print(A)
# print(B)
# print(C)