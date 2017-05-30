import numpy
from scipy.signal import chirp, sawtooth, square, gausspulse
import matplotlib.pyplot as plt

t=numpy.linspace(-1,1,1000)
plt.subplot(221); plt.ylim([-2,2])
plt.plot(t,chirp(t,f0=100,t1=0.5,f1=200))   # plot a chirp
plt.title("Chirp signal")

plt.subplot(222); plt.ylim([-2,2])
plt.plot(t,gausspulse(t,fc=10,bw=0.5))      # Gauss pulse
plt.title("Gauss pulse")

plt.subplot(223); plt.ylim([-2,2])
t*=3*numpy.pi
plt.plot(t,sawtooth(t))                     # sawtooth
plt.xlabel("Sawtooth signal")

plt.subplot(224); plt.ylim([-2,2])
plt.plot(t,square(t))                       # Square wave
plt.xlabel("Square signal")
plt.show()