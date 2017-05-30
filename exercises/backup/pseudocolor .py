import numpy as np
import matplotlib.pyplot as plt

img=plt.imread('stinkbug.png')
lum_img = img[:,:,0]
imgplot = plt.imshow(lum_img)
#imgplot.set_cmap('hot')
imgplot.set_cmap('spectral')
plt.colorbar()
plt.show()
#print(img)