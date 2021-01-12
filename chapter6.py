#Chapter 6 reproduction of material from Prof Krasny's notes Math 471
#Numerical Integration

import matplotlib.pyplot as plt
import numpy as np
import math

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

a = 0
b = 1
n = 8 
h = (b - a) / n
total_area = 0

T_h = 

### *********** END OF RICHARDSON EXTRAPOLATION ************