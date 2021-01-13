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


### ****************** TAYLOR SERIES EXPANSION *********************

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

### ************** END OF TAYLOR SERIES EXPANSION *****************

### ****************** LAGRANGE INTERPOLATION *********************

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

### ************** END OF LAGRANGE INTERPOLATION *****************

### ****************** NEWTON INTERPOLATION **********************

#find a library/function for

### *************** END OF NEWTON INTERPOLATION ******************

### ********************* UNIFORM POINTS *************************

# half_interval_size = 1
# i = 0
# n = 8
# h = 2 / n
# xi = -half_interval_size + (i * h)

# func = 1 / (1 + 25 * x * x)

# x_vals = []
# y_vals = []
# for j in range(n + 1) :
#     x_value = -half_interval_size + (j * h)
#     x_vals.append(x_value)
#     y_value = 1 / (1 + 25 * x_value * x_value)
#     y_vals.append(y_value)

# x_vals2 = np.linspace(-half_interval_size, half_interval_size, 1001)
# y_vals2 = []
# for k in range(1001) :
#     y_value2 = 1 / (1 + 25 * x_vals2[k] * x_vals2[k])
#     y_vals2.append(y_value2)

# coefficients = np.polyfit(x_vals, y_vals, n + 1)
# poly = np.poly1d(coefficients)
# new_x = np.linspace(x_vals[0], x_vals[-1])
# new_y = poly(new_x)
# plt.plot(x_vals, y_vals, "o", new_x, new_y)
# plt.plot(x_vals2, y_vals2)
# plt.ylim(-2, 2)
# plt.show()

## ****************** END OF UNIFORM POINTS *********************

## ******************** CHEBYSHEV POINTS ************************

half_interval_size = 1
i = 0
n = 8
theta = i * math.pi / n
xi = -1 * cos(theta) * half_interval_size

func = 1 / (1 + 25 * x * x)

x_vals = []
y_vals = []
for j in range(n + 1) :
    theta = j * math.pi / n
    x_value = -1 * np.cos(theta) * half_interval_size
    x_vals.append(x_value)
    y_value = 1 / (1 + 25 * x_value * x_value)
    y_vals.append(y_value)

x_vals2 = np.linspace(-half_interval_size, half_interval_size, 1001)
y_vals2 = []
for k in range(1001) :
    y_value2 = 1 / (1 + 25 * x_vals2[k] * x_vals2[k])
    y_vals2.append(y_value2)

coefficients = np.polyfit(x_vals, y_vals, n + 1)
poly = np.poly1d(coefficients)
new_x = np.linspace(x_vals[0], x_vals[-1])
new_y = poly(new_x)
plt.plot(x_vals, y_vals, "o", new_x, new_y)
plt.plot(x_vals2, y_vals2)
plt.ylim(-2, 2)
plt.show()

### ***************** END OF CHEBYSHEV POINTS ********************
