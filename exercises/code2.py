import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats 

# from scipy import linalg
# a = np.random.randn(3, 3) 
# print(a)
# P,D,Q = linalg.svd(a)
# print(np.dot(np.dot(P,np.diag(D)),Q))


# import scipy.misc
# img = scipy.misc.imread('lena.jpg')
# # plt.imshow(img)
# # plt.show()

# import scipy.linalg
# U,s,Vh = scipy.linalg.svd(img)      		
# A = np.dot( U[:,0:32], np.dot(np.diag(s[0:32]),Vh[0:32,:]))

# plt.gray()
# plt.imshow(face)
# plt.show()
# Module 8 Statistics

# a1 = np.random.normal(10,1,30)
# a2 = np.random.normal(9.5,2,30)

# #test = stats.ttest_1samp(a,30)
# test = stats.ttest_ind(a1,a2)
# print(test)

# Module 7 Signal Processing

# N = 800

# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# # y = signal.chirp(x,50,200,1000)
# # plt.plot(x,y)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# plt.plot(x,y)

# from scipy.signal import blackman
# w = blackman(N)
# #w = 1
# yf = fft(y*w)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# plt.figure()
# plt.plot(xf, np.abs(yf[0:N/2]))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Mag')
# plt.grid()
# plt.show()

# from scipy import signal
# t = np.linspace(-1,1,1000)
# s = signal.gausspulse(t,fc=5)
# plt.plot(t,s)
# plt.show()
# t = np.linspace(0,10,1000)
# s = signal.chirp(t,100,200,200)
# plt.plot(t,s)
# plt.show()

# Module 6 Linear Algebra

# import scipy.misc
# img = misc.imread('lena.jpg')
# plt.imshow(img)

# U,s,V = linalg.svd(img)
# print(U)
# print(s)
# print(V)

# plt.figure()
# A = np.dot(U[:,0:32],np.dot())

from scipy import linalg

A = np. matrix([[ 12, -51,   4],
               [  6, 167, -68],
               [ -4,  24, -41]])

print(A)
#Q,R = linalg.qr(A)
# print(Q)
# print(R)
#print(np.dot(Q,R))
P,L,U = linalg.lu(A)
print(np.dot(P,np.dot(L,U)))

#print(U)


# x  + 3y + 5z = 10
# 2x + 5y + z = 8
# 2x + 3y + 8z = 3

# from scipy import linalg

# A = np.matrix("1,3,5;2,5,1;2,3,8")
# b = np.matrix("10,8,3").T
# print(A)
# print(b)
# #x = linalg.solve(A,b)
# x = linalg.lstsq(A,b)
# print(x)

# A = np.matrix("1+2j,3+4j;3+6j,4-8j")
# print(A)
# print(A.H)

#from scipy import integrate

# def dydt(y, t):
#     a = -2.0
#     b = -0.1
#     return y[1],a * y[0] + b * y[1]

# t = np.linspace(0.0, 10.0, 1000)
# yinit = np.array([0, 0.2])

# p = integrate.odeint(dydt,yinit,t)
# plt.plot(t,p[:,0])
# plt.title('Distance vs Time')

# plt.figure()
# plt.plot(t,p[:,1])
# plt.title('Velocity vs Time')
# plt.show()

# def dxdt(x,t):
# 	return np.exp(-x)

# t = np.linspace(0,100,1000)
# x = integrate.odeint(dxdt,0,t)
# plt.plot(t,x)

# plt.figure()
# v = dxdt(x,t)
# plt.plot(t,v)

# plt.show()


# f = lambda x : np.cos(np.exp(x))*np.cos(np.exp(x))
# a = integrate.quad(f,0,3)
# print(a)
# x = np.linspace(0,3,1000)
# y = f(x)
# a = integrate.simps(y,x)
# print(a)
# a = integrate.trapz(y,x)
# print(a)
# f = lambda x: x*x
#a = integrate.quad(f,0,5)
# x = np.linspace(0,5,100)
# y = f(x)
#a = integrate.simps(y,x)
# a = integrate.trapz(y,x)
# print(a)

# def f(a,b,x):
# 	return a*x+b

# x = np.linspace(0,10,100)
# y = x + np.random.randn(100)
# p0,p1 = optimize.curve_fit(f,x,y)
# print(p0,p1)

# y_fit = p0[1]*x+p0[0]

# plt.scatter(x,y)	# raw data
# plt.plot(x,y_fit,'r')	#fitting line
# plt.show()

# from scipy import signal

# p = np.poly1d([1,-3,2])
# print(p)
# print(p.deriv())
# p1 = np.poly1d([1,1,1])
# p2 = np.poly1d([2,1])
# print(p1)
# print(p2)
# print(p1/p2)
# x = np.linspace(-1,1,100)
# y = x*x + 0.03*np.random.randn(100)

# p = np.polyfit(x,y,2)
# p2 = np.poly1d(p,variable='z')
# print(p2)
# plt.scatter(x,y)
# plt.plot(x,p2(x),'r')
# plt.show()

# Module 5: SciPy for Numerical Analysis

# Interpolation
# from scipy import interpolate
# import csv

# a = csv.reader(open('data.csv', newline=''), delimiter=' ')
# x = []
# y = []
# for i in a:
# 	x.append(float(i[0]))
# 	y.append(float(i[1]))
# print(x)
# print(y)
# plt.scatter(x,y)

# x = np.linspace(0,4*np.pi,20)
# y = np.sin(x)
# plt.scatter(x,y)

#f = interpolate.interp1d(x,y,kind='quadratic')
# f = interpolate.UnivariateSpline(x,y,s=1)

# x2 = np.linspace(min(x),max(x),100)
# f2 = f(x2)
# plt.plot(x2,f2,'r')
# plt.show()

# Polynomial
# Create x^2-3x+2

# p = np.poly1d([1,-3,2])
# print(p.r)

# p = np.poly1d([1,2,-8])
# print(p.r)
# print(optimize.fsolve(p,[-3,1]))
# print(optimize.bisect(p,-7,0))
# f = lambda x: np.sin(x)

# print(optimize.fsolve(f,3))
# print(optimize.bisect(f,2,4))

# print(np.roots(p))
#print(np.poly([2,1]))
#print(np.polyval([1,2,3],2))
# with open('test.txt','w') as f:
# 	for i in range(10,20):
# 		f.write('Hello {}\n'.format(i))


# f = open('test.txt','w')

# for i in range(10):
# 	f.write('Hello {}\n'.format(i))

# f.close()

# t = np.linspace(0,10,1000)
# s = signal.chirp(t,100,200,200)
# plt.plot(t,s)
# plt.show()



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

