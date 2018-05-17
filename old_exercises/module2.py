# Numpy and SciPy Essential Training
# Module 2: Numerical Analysis
# Author: Dr. Alfred Ang

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate Noisy Raw data
x = np.linspace(0,10,100)
y = 1.0 + 1.0*x + np.random.normal(0,1,len(x))

# Linear model
def f(a,b,x):
    return a + b*x

# SciPy Curve Fitting 
popt,pcov = curve_fit(f,x,y)
print(popt)

# Plot the raw data and curve fitting
yhat = popt[0]+popt[1]*x
plt.scatter(x,y) #plot the raw data
plt.plot(x,yhat,'r') #plot the curve fitting
plt.show()

# Exercise
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Quadratic model
def f(x,a,b,c): 
    return a + b*x+ c*x*x

# Generate Noisy Raw data
x = np.linspace(-10,10,100) 
y = f(x,1,3,0.5) + 3*np.random.normal(0,1,len(x))

# SciPy Curve Fitting 
popt,pcov = curve_fit(f,x,y)
print(popt)

# Plot the raw data and curve fitting
yhat = popt[0]+popt[1]*x + popt[2]*x*x
plt.scatter(x,y) #plot the raw data
plt.plot(x,yhat,'r') #plot the curve fitting
plt.show()


# Finding Root
import numpy as np
from scipy.optimize import fsolve

f = lambda x : x*x-3*x+2
root = fsolve(f,[-5,5])
print(root)

# Exercise: Find Root
import numpy as np
from scipy.optimize import fsolve

f = lambda x : x*x+2*x-8
root = fsolve(f,[-5,5])
print(root)

# Interpolation
:
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Raw Data
x = np.linspace(0,4*np.pi,20)
y = np.sin(x)


# Interpolation
f = interp1d(x,y,kind="linear")
yhat = f(x)

# Plot out the result
plt.plot(x,y,'bo')
plt.plot(x,yhat,'r')
plt.show()

# Exercise 
]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.linspace(-1,1,10)
y = x*x*x + 0.3*np.random.normal(0,1,10)

# Interpolation
f = interp1d(x,y,kind="cubic")
yhat = f(x)

# Plot out the result
plt.plot(x,y,'bo')
plt.plot(x,yhat,'r')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

x = np.linspace(-1,1,10)
y = x*x*x + 0.3*np.random.normal(0,1,10)

# Interpolation
f = UnivariateSpline(x,y,s=2)
yhat = f(x)

# Plot out the result
plt.plot(x,y,'bo')
plt.plot(x,yhat,'r')
plt.show()


# Integration
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz, quad

x = np.linspace(0,3,100)
f = lambda x : np.cos(np.e**x)*np.cos(np.e**x)

y = f(x)
a1 = simps(y,x)
a2 = trapz(y,x)
a3 = quad(f,0,3)
print('simpson method: ', a1)
print('trapezoid method: ', a2)
print('quadratic method: ', a3)


# ODE
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dxdt = lambda x,t : np.exp(-x)

t = np.linspace(0,100,1000)
x = odeint(dxdt,0,t)

plt.plot(t,x)
plt.xlabel('time')
plt.ylabel('distance')
plt.show()