import numpy
import matplotlib.pyplot as plt

x=numpy.linspace(0,2*numpy.pi,32)
plt.plot(x, numpy.sin(x))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Curve')
plt.show()
