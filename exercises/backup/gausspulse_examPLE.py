import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


t = np.linspace(0,1,1000)
s = signal.gausspulse(t,fc=5)
plt.plot(t,s)
plt.show()



# p = np.poly1d([1,2,3])
# print(p)

# x = np.linspace(0,10,100)
# y = lambda x: x*x-3*x+2

# #a = optimize.fsolve(y,[-1,7])

# #p = np.poly1d([1,2,-8])

# optimize.minimize(y)

# Curve Fitting

# def f(a,b,x):
# 	return a*x+b

# x = np.linspace(0,10,100)
# y = x + np.random.normal(0,1,len(x))

# a,b = optimize.curve_fit(f,x,y)
# y_fit = a[1]*x+a[0]

# plt.scatter(x,y)
# plt.plot(x,y_fit)
# plt.show()

#
#
# a = plt.imread('stinkbug.png')
# plt.imshow(a)
# plt.set_cmap('spectral')
# plt.show()

#Histogram

# x = np.random.randn(1000)
# plt.hist(x)
# plt.show()

#Contour Plot
# x = np.linspace(-1,1,255)
# y = np.linspace(-2,2,300)

# X,Y = np.meshgrid(x,y)

# z = np.sin(X)*np.cos(Y)
# plt.contour(z,255)
# plt.show()
#Bar Plot

# x = np.arange(5)
# y = x
# plt.bar(x,y,color='yellow')
# plt.show()

# Scatter Plot

# x = np.linspace(0,100,200)
# y = x+ 3*np.random.randn(200)
# plt.scatter(x,y)
# plt.show()

# Plot
# def myplot(x,y,i,legend):
# 	plt.subplot(2,2,i)
# 	plt.plot(x,y,label=legend)
# 	plt.grid()
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.axis([0,4*np.pi,-2,2])
# 	plt.legend(loc='upper center')

# x = np.linspace(0,np.pi*4,200)
# y = np.sin(x)
# y2 = np.cos(x)
# y3 = y*y2
# y4 = y**2-y2**2
# myplot(x,y,1,'sin')
# myplot(x,y2,2,'cos')
# myplot(x,y3,3,'sin*cos')
# myplot(x,y4,4,'sin^2-cos^2')
# plt.show()

# Statistics

#print(np.random.randn(5,5))
#print(np.random.normal(5,2,10))

#print(np.random.rand(5,5))
# print(np.random.randint(1,6,5))

# a = np.arange(12).reshape(4,-1)
# print(a)
# print(np.sort(a,axis=0))
#Linear Algebra

# Eigenvalues and Eigenvectors

# A = np.array([[1,0,0],[0,2,0],[0,0,3]])
#print(A)
# w,v = np.linalg.eig(A)
# print(w)
# print(v)

# Solving Linear Equation

# 2x + 3y = 12
# 3x - y = 7

# A = np.matrix([[2,3],[3,-1]])
# b = np.matrix([[12],[7]])

# x = np.linalg.solve(A,b)
#x = A**(-1)*b
# print(x)


# Matrix Multiplication
# a = np.matrix([[1,1],[1,1]])
# b = np.matrix([[2,2],[2,2]])
# print(a*b)



# Logical (Boolean) Indexing/Fancy Indexing

# a = [4,3,7,3,8,10,4,2,9,12]
# b = np.array(a)
# print(sum(b[b%3==0]))

# a = [4,3,-7,3,8,-10,4,2]
# b = np.array(a)

# print(b>0)
# print(b[b>0])
# print(sum(b[b>0]))





# Indexing/ Slicincg

# a = np.arange(24).reshape(4,6)
# print(a)
# print(a[1:3,1:3])
#print(a[:,[1,4]])
# print(a[2:,:])
# print(a[2:,[1,4]])
#print(a[2:,:][:,[1,4]])
# a = np.arange(0,20,2)
# print(a)
# print(a[:8:2])


# import math

# a = math.sin(3)
# print(a)


# a = np.exp(4)
# print(a)

# a = np.array([[1,1],[3,3]])
# b = np.array([2,2])
# a *= 3 
# print(a)
# def f(x,y):
# 	return x**2+y**2

# a = np.fromfunction(f,(4,4))
# print(a)

# a = np.arange(12).reshape(3,4)
# print(a)
# print(a.shape)
# b = a[:,:,np.newaxis]
# print(b)
# print(b.shape)

# Special arrays

# print(np.zeros((2,3)))
# print(np.ones((2,3)))
# print(np.empty((2,3)))
# print(np.full((2,3),5))

# Manipulating the shape

# a = np.array([[1,1],[2,2]])
# b = np.array([[3,3],[4,4]])
# print(a)
# print(b)
# print(np.vstack([a,b]))
# print(np.hstack([a,b]))
# a = np.arange(12)
# print(a.shape)
# b = a[np.newaxis,:]
# print(b.shape)
# a = np.arange(12).reshape(-1,3)
# print(a)
# a.resize((3,4))
# print(a)
# b = a.ravel()
# print(b)
# Create special sequence

#a = np.linspace(0,10,20)
# a = np.arange(0,10,2)
# print(a)
# print(len(a))

# Create Numpy Array from Python List
#a = [[1,1,1],[2,2,2]]
#a1 = np.array(a,dtype=np.float64)
# print(a1)
# print(a1.dtype)
# print(a1.shape)
# c = a1.astype(np.int64)
# print(c.dtype)
#print(b1.dtype)


# import pandas as pd 
# a2 = pd.Series(a)
# b2 = pd.Series(b)
# print(a2)
# print(b2)

