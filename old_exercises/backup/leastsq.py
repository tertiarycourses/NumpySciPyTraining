import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
y = x*x + 0.03*np.random.normal(0,1,100)

p = np.polyfit(x,y,2)
p2 = np.poly1d(p)
y_fit = np.polyval(p,x)
plt.scatter(x,y)
plt.plot(x,y_fit,'r')
plt.show()


# plt.show()

# import numpy as np

# A = np.array([[1,3,5],[2,5,1],[2,3,8]])
# b = np.array([[10],[8],[3]])

# #solution=np.linalg.lstsq(A,b) 

# solution= linalg.inv(A).dot(b)
# print (solution[0])  