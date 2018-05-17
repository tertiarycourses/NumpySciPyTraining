import numpy

a=numpy.zeros((5,5), dtype=int)
b=numpy.ones((2,2), dtype=int)
c=numpy.identity(3, dtype=int)
d=numpy.eye(4,k=1) + numpy.eye(4,k=-1)
print(a)
print(b)
print(c)
print(d)
