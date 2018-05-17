import numpy as np
from scipy import integrate

f = lambda x: x*x

x = np.linspace(0,5,100)
y = f(x)
#t = integrate.simps(y,x)
t = integrate.trapz(y,x)
print(t)