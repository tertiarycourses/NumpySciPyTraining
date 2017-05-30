import numpy as np


a = np.array([1,2,3,4])
b = a.copy()
a = a[::-1]
c = a + b
d = a - b
print 'a = ', a 
print 'b = ', b
print 'c = ', c
print 'd = ', d