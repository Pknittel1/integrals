# #Chapter 5 reproduction of material from Prof Krasny's notes Math 471
# #Interpolation

from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# Define symbol
x = Symbol('x')


### ****************** TAYLOR SERIES EXPANSION **********************************************************

# plt.rcParams['lines.linewidth'] = 2

# # Function for Taylor Series Expansion
# def taylor(function, x0, n):
#     """
#     Parameter "function" is our function which we want to approximate
#     "x0" is the point where to approximate
#     "n" is the order of approximation
#     """
#     return function.series(x,x0,n).removeO()

# func = 1 / (1 + 25 * x * x)

# # This will plot sine and its Taylor approximations
# p = plot(func, taylor(func, 0, 1), taylor(func, 0, 3), taylor(func, 0, 5),
#          (x, -0.3, 0.3),legend=True, show=False)

# p[0].line_color = 'blue'
# p[1].line_color = 'green'
# p[2].line_color = 'firebrick'
# p[3].line_color = 'cyan'
# p.title = 'Taylor Series Expansion'
# p.show()

### ************** END OF TAYLOR SERIES EXPANSION ******************************************************

### ****************** LAGRANGE INTERPOLATION **********************************************************

# does not work right now

#interpolate f(x)=x^3 with three points
#lagrange polynomial thus has degree 2 since 3 points
# x = np.array([-1, 0, 1])
# y = x**3
# poly = lagrange(x, y)
# Polynomial(poly).coef

# print(poly)
# print(Polynomial(poly).coef)

# p = plot(y, Polynomial(poly).coef, (x,-0.2,0.2),legend=True, show=False)
# p[0].line_color = 'blue'
# p[1].line_color = 'green'

# p.show()

### ************** END OF LAGRANGE INTERPOLATION ******************************************************

### ********************* UNIFORM POINTS **************************************************************
# L = 1
# i = 0
# n = 8
# h = 2 / n
# xi = -L + (i * h)

# #finding the uniform points of the given function
# x_vals = []
# y_vals = []
# for j in range(n + 1) :
#     x_value = -L + (j * h)
#     x_vals.append(x_value)
#     y_value = 1 / (1 + 25 * x_value * x_value)
#     #y_value = np.exp(-(x_value * x_value))
#     y_vals.append(y_value)

# # plotting the actual function
# x_vals2 = np.linspace(-L - 0.5, L + 0.5, 1001)
# y_vals2 = []
# for k in range(0, 1001) :
#     y_value2 = 1 / (1 + 25 * x_vals2[k] * x_vals2[k])
#     #y_value2 = np.exp(-(x_vals2[k] * x_vals2[k]))
#     y_vals2.append(y_value2)

# def _poly_newton_coefficient(x, y):
#     m = len(x)
#     x = np.copy(x)
#     a = np.copy(y)
#     for k in range(1, m):
#         a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
#     return a

# def newton_polynomial(x_data, y_data, x):
#     a = _poly_newton_coefficient(x_data, y_data)
#     n = len(x_data) - 1  # Degree of polynomial
#     p = a[n]
#     for k in range(1, n + 1):
#         p = a[n - k] + (x - x_data[n - k])*p
#     return p

# # plotting the newton polynomial
# x_vals3 = np.linspace(-L - 0.5, L + 0.5, 1001)
# y_vals3 = []
# for k in range(0, 1001) :
#     y_value3 = (newton_polynomial(x_vals, y_vals, x_vals3[k]))
#     y_vals3.append(y_value3)

# # plotting the cehbyshev points
# plt.plot(x_vals, y_vals, "o")
# # plotting the actual function
# plt.plot(x_vals2, y_vals2)
# # plotting newton polynomial
# plt.plot(x_vals3, y_vals3)
# plt.ylim(-0.5, 1.5)
# plt.xlabel('y')
# plt.ylabel('x')
# plt.title('he')
# plt.title('Newton Polynomial with Uniform Points, N = {}'.format(n))
# plt.show()

## ****************** END OF UNIFORM POINTS **********************************************************

## ******************** CHEBYSHEV POINTS NEWTON METHOD **************************************************

L = 1
i = 0
n = 4
theta = i * math.pi / n
xi = -1 * cos(theta) * L

#finding the chebyshev points of the given function
x_vals = []
y_vals = []
for j in range(n + 1) :
    theta = j * math.pi / n
    x_value = -1 * np.cos(theta) * L
    x_vals.append(x_value)
    y_value = 1 / (1 + 25 * x_value * x_value)
    #y_value = np.exp(-(x_value * x_value))
    y_vals.append(y_value)

# plotting the actual function
x_vals2 = np.linspace(-L - 0.5, L + 0.5, 1001)
y_vals2 = []
for k in range(0, 1001) :
    y_value2 = 1 / (1 + 25 * x_vals2[k] * x_vals2[k])
    #y_value2 = np.exp(-(x_vals2[k] * x_vals2[k]))
    y_vals2.append(y_value2)

# creating polynomial, returns 
def _poly_newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        # a[start:stop] = a[k:m], same notation as notes
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])
    return a

# returns pn(x) evaluted at x
def newton_polynomial(x_data, y_data, x):
    a = _poly_newton_coefficient(x_data, y_data)
    # Degree of polynomial
    n = len(x_data) - 1 
    p = a[n]
    for k in range(1, n + 1):
        #updating pn(x) polynomial based on pn-1(x) and extra factor
        p = a[n - k] + (x - x_data[n - k]) * p
    print(p)
    return p

# plotting the newton polynomial
x_vals3 = np.linspace(-L - 0.5, L + 0.5, 1001)
y_vals3 = []
for k in range(0, 1001) :
    y_value3 = (newton_polynomial(x_vals, y_vals, x_vals3[k]))
    y_vals3.append(y_value3)

# plotting the cehbyshev points
plt.plot(x_vals, y_vals, "o")
# plotting the actual function
plt.plot(x_vals2, y_vals2)
# plotting newton polynomial
plt.plot(x_vals3, y_vals3)
plt.ylim(-0.5, 1.5)
plt.xlabel('y')
plt.ylabel('x')
plt.title('he')
plt.title('Newton Polynomial with Chebyshev Points, N = {}'.format(n))
plt.show()

### ***************** END OF CHEBYSHEV POINTS ********************
