import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.linspace(0,np.pi*4,20)
y = np.sin(x)
plt.scatter(x,y)
# plt.show()

#f = interpolate.interp1d(x,y,kind="linear")
f = interpolate.interp1d(x,y,kind="quadratic")

x2 = np.linspace(x.min(), x.max(), 1000)
f2 = f(x2)
plt.plot(x2,f2)
plt.show()

# polynomial=scipy.interpolate.lagrange(x, y)

# xn = scipy.linspace(0,np.pi/2,100)
# plt.plot(xn,polynomial(xn))
# plt.plot(x,y,'or')
# plt.show()