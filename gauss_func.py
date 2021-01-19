#Exploring integrals of different gaussian distributions

from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from mpl_toolkits import mplot3d

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 2D
# simplification of integrating over real numbers by assuming func is zero beyond +/-10

# #func = e^-x^2
# f = lambda x : np.exp(-x**2)
# a = -10; b = 10
# exact_area = np.sqrt(np.pi)
# print("exact area  =", exact_area)

# arr = [2, 4, 8, 16, 32, 64]
# for n in arr :
#     h = (b - a) / n
#     total_area = 0
#     if n > 2:
#         for j in range(1, n) :
#             total_area += f(a + j * h)
#         total_area += 0.5 * f(a)
#         total_area += 0.5 * f(b)
#         total_area *= h
#     else:
#         total_area += h * 0.5 * (f(a) + f(a + h))
#         total_area += h * 0.5 * (f(a + h) + f(a + 2*h))
#     print("area approximation =", total_area)
#     print("error =", np.abs(total_area - exact_area))

# N = 64
# # x and y values for the trapezoid rule
# x = np.linspace(a,b,N+1)
# y = f(x)
# # X and Y values for plotting y=f(x)
# X = np.linspace(a,b,100)
# Y = f(X)
# plt.plot(X,Y)
# for i in range(N):
#     xs = [x[i],x[i],x[i+1],x[i+1]]
#     ys = [0,f(x[i]),f(x[i+1]),0]
#     plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)
# plt.title('Uniform Trapezoid Rule, N = {}'.format(N))
# plt.show()

# TRAPEZOID INTEGRATION WITH NON-UNIFORM POINT DISTRIBUTION 2D

#func = e^-x^2
f = lambda x : np.exp(-x**2)
a = -10; b = 10
exact_area = np.sqrt(np.pi)
print("exact area  =", exact_area)

arr = [2, 4, 8, 16, 32, 64]
for n in arr :
    h = (b - a) / n
    total_area = 0
    if n > 2:
        for j in range(1, n) :
            total_area += f(a + j * h)
        total_area += 0.5 * f(a)
        total_area += 0.5 * f(b)
        total_area *= h
    else:
        total_area += h * 0.5 * (f(a) + f(a + h))
        total_area += h * 0.5 * (f(a + h) + f(a + 2*h))
    print("area approximation =", total_area)
    print("error =", np.abs(total_area - exact_area))

N = 64
# x and y values for the trapezoid rule
x = np.linspace(a,b,N+1)
y = f(x)
# X and Y values for plotting y=f(x)
X = np.linspace(a,b,100)
Y = f(X)
plt.plot(X,Y)
for i in range(N):
    xs = [x[i],x[i],x[i+1],x[i+1]]
    ys = [0,f(x[i]),f(x[i+1]),0]
    plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)
plt.title('Non-Uniform Trapezoid Rule, N = {}'.format(N))
plt.show()

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 3D
# simplification of integrating over real numbers by assuming func is zero beyond +/-10

# #func = e^-(x^2 + y^2)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# f = lambda x, y : np.exp(-(x**2 + y**2))
# a = -5; b = 5; N = 30
# #exact_area = np.sqrt(np.pi)
# #print("exact area  =", exact_area)

# arr = [4, 8, 16, 32, 64]
# for n in arr :
#     h = (b - a) / n
#     total_area = 0
#     if n > 2:
#         for j in range(0, n) :
#             for k in range(1, n) :
#                 total_area += h * 0.5 * f(k, j)
#                 total_area += h * 0.5 * f(k, j + 1)
#             total_area += h * 0.25 * f(0, j)
#             total_area += h * 0.25 * f(0, j + 1)
#             total_area += h * 0.25 * f(n, j)
#             total_area += h * 0.25 * f(n, j + 1)
#     #else:
#         #fix
#     print("area approximation =", total_area)
#     #print("error =", np.abs(total_area - exact_area))

# # x and y values for the trapezoid rule
# x = np.linspace(a, b, N+1)
# y = np.linspace(a, b, N+1)
# z = f(x, y)
# # X and Y values for plotting z = f(x, y)
# X = np.linspace(a, b, 100)
# Y = np.linspace(a, b, 100)
# Z = f(X, Y)
# plt.plot(X, Y, Z)

# for i in range(N):
#     for j in range(N):
#         xs = [x[i], x[i], x[i+1], x[i+1]]
#         ys = [y[j], y[j], y[j+1], y[j+1]]
#         zs = [0, f(x[i], y[j]), f(x[i+1], y[j+1]), 0]
#         ax.plot3D(xs, ys, zs, 'red')

# plt.title('Uniform Trapezoid Rule, N = {}'.format(N))
# plt.show()