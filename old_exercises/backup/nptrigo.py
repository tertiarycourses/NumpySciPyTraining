import numpy
import matplotlib.pyplot as plt

x=numpy.linspace(0,2*numpy.pi,32)
sin = numpy.sin(x)
cos = numpy.cos(x)
tan = numpy.tan(x)

print(x)
print('sin(x) =', sin)
print('cos(x) =', cos)
print('tan(x) =', tan)