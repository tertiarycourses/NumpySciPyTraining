import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


#x = np.linspace(0,10,100)
p = lambda x: x*x+2*x-8

a = optimize.fsolve(p,[-1,7])
print(a)

# def f(x,a,b):
# 	return x*x+a*x+b

# x = np.linspace(-10,10,100)
# y = f(x,2,1) + 3*np.random.normal(0,1,len(x))

# a,b = optimize.curve_fit(f,x,y)
# y_fit = f(x,a[0],a[1])
# plt.scatter(x,y)
# plt.plot(x,y_fit)
# plt.show()