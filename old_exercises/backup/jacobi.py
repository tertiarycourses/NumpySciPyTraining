import numpy as np 
import scipy.special
import matplotlib.pyplot as plt

x=np.linspace(-1,1,1000)
plt.plot(x,scipy.special.eval_jacobi(3,0,1,x))
plt.show()