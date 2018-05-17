import numpy as np

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.cross(x, y)
print(z)