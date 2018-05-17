
import numpy as np

P1 = np.poly1d([1,0,1])           
P2 = np.poly1d([2,1])
P3 = P1/P2
print(P3)