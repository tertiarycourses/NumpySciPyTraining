import numpy as np
from scipy import stats 

a = np.random.normal(5,1,1000)
b = np.random.normal(4.9,1,1000)
test = stats.ttest_ind(a,b)
print(test)