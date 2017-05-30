# Distribution Fitting

import numpy
data = numpy.array([[113,105,130,101,138,118,87,116,75,96, \
             122,103,116,107,118,103,111,104,111,89,78,100,89,85,88], \
         [137,105,133,108,115,170,103,145,78,107, \
              84,148,147,87,166,146,123,135,112,93,76,116,78,101,123]])

dataDiff = data[1,:]-data[0,:]

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 5.0)

from scipy.stats import norm     # Gaussian distribution
mean,std=norm.fit(dataDiff)

plt.hist(dataDiff, normed=1)
x=numpy.linspace(dataDiff.min(),dataDiff.max(),1000)
pdf=norm.pdf(x,mean,std)
plt.plot(x,pdf)
plt.show()