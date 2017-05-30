import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[0.5,4.5])
ax.set(ylim=[-2,8])
ax.set(title="An Example Axis")
ax.set(ylabel='Y-axis')
ax.set(xlabel='X-axis')
plt.show()