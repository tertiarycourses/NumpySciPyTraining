import numpy as np
import scipy.integrate


def func(x,a,b):
	return a*x**b

a, b = 3,2

result = scipy.integrate.quad(func, 0, 2, args=(a,b))
print(result[0])