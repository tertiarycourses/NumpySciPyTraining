# Numpy and SciPy Essential Training
# Module 5: Signal Processing
# Author: Dr. Alfred Ang

from scipy.signal import chirp, gausspulse
from scipy.fftpack import fft
from scipy.signal import blackman
import matplotlib.pyplot as plt 
import numpy as np

# Waveforms
# t = np.linspace(0,10,1000)
# s = gausspulse(t,fc=5)
# s = chirp(t,100,200,200)
# plt.plot(t,s)
# plt.show()

# FFT
N = 1024
T = 1/N

t = np.linspace(0,N*T,N)
y = np.sin(2*np.pi*50*t)+0.2*np.sin(2*np.pi*80*t)+0.5*np.sin(2*np.pi*120*t)
w = blackman(N)

yf = fft(y*w)
f = np.linspace(0,1/(2*T),N/2)

#plt.plot(t,y)
plt.semilogy(f,np.abs(yf[0:512]))
plt.show()
