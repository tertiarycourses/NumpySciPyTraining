# Linear Algebra
from scipy.linalg import solve,lstsq
from scipy.linalg import qr,lu,svd
import numpy as np

# A = np.matrix([
# 	[2,3],
# 	[3,-1]])

# y = np.matrix([[12],[7]])

# x = solve(A,y)
# #x = lstsq(A,y)
# print(x)

A = np.matrix([
	[12,-51,4],
	[6,167,-68],
	[-4,24,41]])

# A = np.matrix([
# 	[2,0,0],
# 	[0,3,0],
# 	[0,0,5]])

# Q,R = qr(A)
# print(Q)
# print(R)
# B = np.dot(Q,R)
# print(A)
# print(B)

# P,L,U = lu(A)
# print(P)
# print(L)
# print(U)
# B = np.dot(np.dot(P,L),U)
# print(A)
# print(B)

P,D,Q = svd(A)
# print(P)
# print(D)
# print(Q)
B = np.dot(np.dot(P,np.diag(D)),Q)
print(A)
print(B)