import numpy as np


y= np.poly1d([3,0,1]) 
print(y(2))

z = np.polyval(y,2)
print(z)
