import scipy.misc
import matplotlib.pyplot as plt 

img=scipy.misc.lena()
plt.gray() 
plt.imshow(img)
plt.show() 