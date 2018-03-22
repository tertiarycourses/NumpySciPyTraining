# Numpy and SciPy Essential Training
# Module 3: Linear Algebra
# Author: Dr. Alfred Ang

from scipy.linalg import solve,lstsq
from scipy.linalg import qr,lu,svd
import numpy as np

# Solving Linear Equations
# A = np.matrix([
# 	[2,3],
# 	[3,-1]])

# y = np.matrix([[12],[7]])

# x = solve(A,y)
# #x = lstsq(A,y)
# print(x)

# Matrix Decomposition
A = np.matrix([
	[12,-51,4],
	[6,167,-68],
	[-4,24,41]])

# A = np.matrix([
# 	[2,0,0],
# 	[0,3,0],
# 	[0,0,5]])

# QR Decomposition
# Q,R = qr(A)
# print(Q)
# print(R)
# B = np.dot(Q,R)
# print(A)
# print(B)

# LU Decomposition
# P,L,U = lu(A)
# print(P)
# print(L)
# print(U)
# B = np.dot(np.dot(P,L),U)
# print(A)
# print(B)

# SVD Decomposition
P,D,Q = svd(A)
# print(P)
# print(D)
# print(Q)
B = np.dot(np.dot(P,np.diag(D)),Q)
print(A)
print(B)