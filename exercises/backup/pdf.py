import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x),'r-', lw=5)

r = norm.rvs(size=1000)
plt.hist(r, normed=True, histtype='stepfilled',alpha=0.2)
plt.show()