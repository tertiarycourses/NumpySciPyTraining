import numpy as np

a = np.array([3,7,5,10])
b = a[::-1]
c = np.sort(a)
d = a.copy()
e = a.sum()
d[0] = 123
print(a)
print(b)
print(c)
print(d)
print(e)
