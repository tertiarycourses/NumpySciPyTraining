from scipy.stats import norm     # Gaussian distribution
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage

plt.gray()
lena=scipy.misc.lena().astype(float)

plt.subplot(221);
plt.imshow(lena)
lena+=norm(loc=0,scale=16).rvs(lena.shape)

plt.subplot(222);
plt.imshow(lena)
denoised_lena = scipy.ndimage.median_filter(lena,3)

plt.subplot(224); 
plt.imshow(denoised_lena)

from scipy.ndimage.filters import sobel
import numpy
lena=scipy.misc.lena()
sblX=sobel(lena,axis=0); sblY=sobel(lena,axis=1)
sbl=numpy.hypot(sblX,sblY)

plt.subplot(223); 
plt.imshow(sbl) 
plt.show()