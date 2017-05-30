import numpy as np

#a = np.array([[0,1],[1,0]])
#a = np.mat(a)
#b = np.array([[1,2],[2,1]])
#b = np.mat(b)

a = np.matrix([[0,1],[1,0]])
b = np.matrix([[1,2],[2,1]])

c = a * b
d = np.multiply(a,b)
print(a)
print(b)
print(c)
print(d)