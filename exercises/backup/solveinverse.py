import numpy as np

A = np.array([[1,3,5],[2,5,1],[2,3,8]])
b = np.array([[10],[8],[3]])

solution= np.linalg.inv(A).dot(b)
print (solution)  