#Exploring integrals of different gaussian distributions

from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION

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
            total_area += math.e ** (-((a + j * h) ** 2))
        total_area += 0.5 * math.e ** (-((a + 0 * h) ** 2))
        total_area += 0.5 * math.e ** (-((a + n * h) ** 2))
        total_area *= h
    else:
        total_area += h * 0.5 * (math.e ** (-(a ** 2)) + math.e ** (-((a + h) ** 2)))
        total_area += h * 0.5 * (math.e ** (-((a + h) ** 2)) + math.e ** (-((a + 2 * h) ** 2)))
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
plt.title('Uniform Trapezoid Rule, N = {}'.format(N))
plt.show()

# TRAPEZOID INTEGRATION WITH NON-UNIFORM POINT DISTRIBUTION

# #func = e^-x^2
# f = lambda x : np.exp(-x**2)
# a = -5; b = 5; N = 30

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

# plt.title('Non-Uniform Trapezoid Rule, N = {}'.format(N))
# plt.show()