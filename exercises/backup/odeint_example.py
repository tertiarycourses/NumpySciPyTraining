import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# def dxdt(x,t):
# 	return np.exp(-x)

# t = np.linspace(0,100,1000)
# x = integrate.odeint(dxdt,0,t)
# plt.plot(t,x)
# plt.show()


def dydt(y, t):
    a = -2.0
    b = -0.1
    return y[1],a * y[0] + b * y[1]

t = np.linspace(0.0, 10.0, 1000)
yinit = np.array([0.0005, 0.2])
y = integrate.odeint(dydt, yinit, t)
plt.plot(t, y[:, 0])
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# def dydt(y, t):
#     a = -2.0
#     b = -0.1
#     return np.array([y[1], a * y[0] + b * y[1]])

# time = np.linspace(0.0, 10.0, 1000)
# yinit = np.array([0.0005, 0.2])
# y = odeint(deriv, yinit, time)
# plt.plot(time, y[:, 0])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
