import numpy as np
from scipy import stats 

a = np.random.normal(5,1,1000)
test = stats.ttest_1samp(a,5)
print(test)