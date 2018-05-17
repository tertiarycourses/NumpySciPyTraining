import numpy as np
scores = np.array([114, 100, 104, 89, 102, 91, 114, 114, 103, 105, 108, 130, 120, 132, 111, 128, 118, 119, 86, 72, 111, 103, 74, 112, 107, 103, 98, 96, 112, 112, 93])

mean = np.mean(scores)
std = np.std(scores)

print("Mean of score is ", mean)
print("Standard Deviation of scores is ", std)
