import numpy as np

# Finding Roots
p = np.poly1d([1,-3,2])
print(p)
print(p.r)

p = np.roots([1,-3,2])
print(p)

#Determine the coeficient
p = np.poly([2,1])
print(p)

# Creating Polynomial
p = np.poly1d([1,-3,2])
print(p(0.5))
p = np.polyval([1,2,3],0.5)
print(p)


# P1=np.poly1d([1,-3,2])           # using coefficients
# print (P1.r); print (P1.o); print (P1.deriv())