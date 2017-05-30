import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


t = np.linspace(0,10,1000)
s = signal.chirp(t,100,200,200)
plt.plot(t,s)
plt.show()
