import numpy as np
from scipy import integrate

f = lambda x: x*x

t = integrate.quad(f,0,5)
print(t)

