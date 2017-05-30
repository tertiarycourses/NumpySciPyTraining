import numpy as np
#from scipy import linalg

A = np.array([[1,3,5],[2,5,1],[2,3,8]])
b = np.array([[10],[8],[3]])
solution = np.linalg.solve(A,b)

print(solution)