import numpy as np
import scipy
import matplotlib.pyplot as plt


x=np.linspace(0,np.pi/2,10)
y=np.sin(x)
line=np.polyfit(x,y,deg=1)

plt.plot(x,y,'or')
plt.plot(x,np.polyval(line,x),'r')

plt.show()