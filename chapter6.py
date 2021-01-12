#Chapter 6 reproduction of material from Prof Krasny's notes Math 471
#Numerical Integration

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.integrate as integrate
import scipy.special as special
from sympy import series, Symbol

# Define symbol
x = Symbol('x')

### ****************** TRAPEZOID RULE *********************

# a = 0
# b = 1
# n = 8 
# h = (b - a) / n
# total_area = 0
# #func = math.e ** (-(x ** 2))

# if n > 2:
#     for j in range(1, n) :
#         total_area += math.e ** (-((a + j * h) ** 2))
#     total_area += 0.5 * math.e ** (-((a + 0 * h) ** 2))
#     total_area += 0.5 * math.e ** (-((a + n * h) ** 2))
#     total_area *= h
# else:
#     total_area += 0.5 * math.e ** (-((a + 0 * h) ** 2))
#     total_area += 0.5 * math.e ** (-((a + n * h) ** 2))

# print("area approximation =", total_area)

# x_vals = np.linspace(-2, 2, 1001)
# y_vals = []
# for k in range(1001) :
#     y_value = math.e ** (-(x_vals[k] ** 2))
#     y_vals.append(y_value)
# plt.plot(x_vals, y_vals)
# plt.show()

### *************** END OF TRAPEZOID RULE ******************

### ************** RICHARDSON EXTRAPOLATION ****************

# a = 0
# b = 1
# n = 8 
# h = (b - a) / n
# total_area = 0

# T_h = 

### *********** END OF RICHARDSON EXTRAPOLATION ************

### *************** ORTHOGONAL POLYNOMIALS *****************

# vec1 = [1, 1]
# vec2 = [1, 2]

# if len(vec1) == len(vec2):
#     inner_product = 0
#     for i in range(len(vec2)):
#         inner_product += vec1[i] * vec2[i]
#     print(inner_product)

# def f(x):
#     return x * x
# def g(x):
#     return x 
# # <f,g> notation
# result = integrate.quad(lambda x: f(x) * g(x), -1, 1)
# print(result)

### ************ END OF ORTHOGONAL POLYNOMIALS *************

### ***************** GAUSSIAN QUADRATURE ******************

# f1 = 1
# def f2(x):
#     return x 
# def f3(x):
#     return x * x

# # <f,g> notation
# def func1(fx, gx):
#     return integrate.quad(lambda x: fx * gx, -1, 1)

# # ||f|| = sqrt(<f,f>)
# def func2(fx):
#     return sqrt(integrate.quad(lambda x: fx * fx, -1, 1))

# P0 = f1
# P1 = f2(x) + f2(x) * func1(f2(x), f1) / f1
# print(P1)
# #P2 =

### ************ END OF GAUSSIAN QUADRATURE ****************

### ************** GAUSS-LAGUERRE QUADRATURE ***************



### ********** END OF GAUSS-LAGUERRE QUADRATURE ************