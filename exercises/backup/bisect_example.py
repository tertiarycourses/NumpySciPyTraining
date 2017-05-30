import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

x = np.linspace(0,10,100)
y = lambda x: x*x-3*x+2

#a = optimize.fsolve(y,[-1,7])

#p = np.poly1d([1,2,-8])

print(optimize.bisect(y, -1,1.5))
