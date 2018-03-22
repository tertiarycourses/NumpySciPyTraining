# Numpy and SciPy Essential Training
# Module 1: Basics of Numpy
# Author: Dr. Alfred Ang

import numpy as np 
import matplotlib.pyplot as plt

# Create 1D Numpy Array
# a = [1.1,2,3]
# b = [3,2,1]
# print(a+b)

# a1 = np.array(a,dtype=np.float32)
# b1 = np.array(b,dtype=np.float32)
# print(a1-b1)

# Create 2D Numpy Array
# a = [
# 	[1,2],
# 	[3,4]]
# b = [
# 	[4,3],
# 	[2,1]]

# b = [
# 	[4,3],
# 	[2,1],
# 	[5,6]]

# a1 = np.array(a,dtype=np.float32)
# b1 = np.array(b,dtype=np.float64)
# b1 = b1.astype(np.float32)

# Array Attributes
# print(a1.shape)
# print(b1.ndim)
# print(len(b1))
# print(b1.shape)
# c1 = a1+b1
# print(c1.dtype)

# Create Sequence
# a = np.linspace(1,10,100,dtype=np.float32)
# print(a.dtype)
# print(a)

# a = np.arange(1,10,2)
# print(a)

# Reshape
# a = np.linspace(1,12,12)
# b = a.reshape((-1,6))
# print(a)
# print(b)

# a = np.linspace(1,10,10)
# b = a[np.newaxis,:]
# c = b.ravel()
# print(a)
# print(b)
# print(c)
# print(a.ndim)
# print(b.ndim)
# print(a.shape)
# print(b.shape)
# print(c.shape)

# a = np.linspace(1,12,12).reshape((2,6))
# print(a)

# Quiz
# a = np.linspace(1,24,24).reshape((2,6,2))
# print(a)
# print(a.shape)
# print(a.ndim)

# c = b[np.newaxis,:]
# c = b[:,np.newaxis]
# print(c)
# print(c.shape)
# print(c.ndim) 

# Exercise
# a = np.linspace(1,12,12)
# b = a.reshape((-1,6))
# print(b)
# a = b.ravel()
# print(a)
# print(a.shape)

# Stacking array
# a = np.linspace(1,6,6).reshape((2,3))
# b = np.linspace(7,12,6).reshape((2,3))
# c = np.vstack([a,b])
# d = np.hstack([a,b])
# print(c)
# print(d)

# Special Matrices
# print(np.zeros((3,2)))
# print(np.ones((3,2)))
# print(np.diag([1,2,3]))
# print(np.full((3,3),1))

# Create matrices using your own function
# def f(x,y):
# 	return 10*x+y
# a = np.fromfunction(f,(3,3))

# a = np.fromfunction(lambda x,y:10*x+y,(3,3))
# print(a)


# Slicig 1D Numpy Array
# a = np.linspace(1,10,10)
# print(a)
# print(a[1:4])
# print(a[:4])
# print(a[4:])
# print(a[:])
# print(a[-1])
# print(a[-3:-1])

# Slicig 2D Numpy Array
# a = np.linspace(1,16,16).reshape((4,4))
# print(a)
# b = a[:,[1,3]]
# print(b[[1,3],:])

# Logical Indexing
# a = np.linspace(1,16,16).reshape((4,4))
# print(a)
# print(a[a%3==0])
# print(a[a>7])

# Polynomial

# a = np.poly1d([1,2,-3])

# Evaluate polynomial
# print(np.polyval(a,2))

# Finding root
# print(a.r)

# Derviative
# b = a.deriv()
# print(b)


# Poly fit
# x = np.linspace(-10,10,100)
# y = x*x+10*np.random.random(len(x))
# p = np.polyfit(x,y,2)
# print(p)
# py = np.polyval(p,x)

# plt.plot(x,y,'o')
# plt.plot(x,py,'r')
# plt.show()

# x = np.linspace(-1,1,100)
# y = x*x*x + 0.03*np.random.normal(0,1,100)

# p = np.polyfit(x,y,3)
# # print(p)
# py = np.polyval(p,x)

# plt.plot(x,y,'o')
# plt.plot(x,py,'r')
# plt.show()

# Linear Algebra
# a = np.array([
# 	[1,2],
# 	[3,4]])

# a = np.matrix([
# 	[1,2],
# 	[3,4]])

# b = np.array([
# 	[1,-1],
# 	[-1,1]])

# b = np.matrix([
# 	[1,-1],
# 	[-1,1]])

# Matrix Multiplicatoin
# print(a*b)
# print(np.dot(a,b))

# Solving Linear Equation
# A = np.matrix([
# 	[2,3],
# 	[3,-1]])

# y = np.matrix([[12],[7]])

# x = np.linalg.solve(A,y)
# print(x)


# Challenge
# A = np.matrix([
# 	[3,6,-5],
# 	[1,-3,2],
# 	[5,-1,4]])

# y = np.matrix([
# 	[12],
# 	[-2],
# 	[10]])

# x = np.linalg.solve(A,y) 

# print(x)
# Eigenvalues and Eigenvectors
# A = np.matrix([
# 	[1,0,0],
# 	[0,2,0],
# 	[0,0,3]])

# w,v = np.linalg.eig(A)
# print(w)

# Statistics
# A = np.array([
# 	[3,6,-5],
# 	[1,-3,2],
# 	[5,-1,4]])

# print(np.mean(A,axis=1))

# Random normal

# print(np.random.normal(0,1,[2,3]))
# print(np.random.rand(10))