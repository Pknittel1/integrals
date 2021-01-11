# #Chapter 5 reproduction of material from Prof Krasny's notes Math 471
# #Interpolation

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# # Data for plotting
# x = np.arange(-4.0, 4.0, 0.01)
# y = 2 ** x
# #y = 1 / (1 + 25 * x * x)
# n = 1
# #polynomial estimate function
# p = 0

#fig, ax = plt.subplots()
# ax.plot(x, y)

# plt.rcParams['figure.figsize'] = 13,10
# plt.rcParams['lines.linewidth'] = 2
# x = Symbol('x')
# def taylor(function, x0, n):
#     """
#     Parameter "function" is our function which we want to approximate
#     "x0" is the point where to approximate
#     "n" is the order of approximation
#     """
#     return function.series(x,x0,n).removeO()
# print('x^3 =', taylor(x*x*x, 0, 4))

#for i in range(1, n + 1): 

# x_vals = []
# y_vals = []

# x = 2
# for i in range(-4, 5):
#     e_to_2 = 0
#     for j in range(8):
#         e_to_2 += (2**x)**j/math.factorial(j)
#     x_vals.append(i)
#     y_vals.append(e_to_2)

# #if you get rid of the ", 'ro", then it'll form a line
# plt.plot(x_vals, y_vals, 'ro')
# plt.plot(2.741828**2, 5)

# plt.plot()

# ax.set(title='Example 1')
# ax.grid()
# fig.savefig("test.png")
# plt.show()


# how to approximate e^x at x = 2
# x = 2
# e_to_2 = x**0/math.factorial(0) + x**1/math.factorial(1) + x**2/math.factorial(2) + x**3/math.factorial(3) + x**4/math.factorial(4)
# print(e_to_2)






from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

plt.rcParams['figure.figsize'] = 13,10
plt.rcParams['lines.linewidth'] = 2

x = Symbol('x')


def taylor(function, x0, n):
    """
    Parameter "function" is our function which we want to approximate
    "x0" is the point where to approximate
    "n" is the order of approximation
    """
    return function.series(x,x0,n).removeO()

x1 = np.arange(-4.0, 4.0, 0.01)
sin = taylor(sin(x), 0, 4)
# print(sin)

# x_vals = []
# y_vals = []

# x = 2
# for i in range(-4, 5):
#     x_vals.append(i)
#     y_vals.append(taylor(sin(x), 0, 4).subs((x,i)

# #if you get rid of the ", 'ro", then it'll form a line
# plt.plot(x_vals, y_vals)

# fig, ax = plt.subplots()
# ax.plot(x1, taylor(sin(x), 0, 4).subs(x1))
p = plot(sin(x),taylor(sin(x),0,1),taylor(sin(x),0,3),taylor(sin(x),0,5),
         (x,-3.5,3.5),legend=True, show=False)

plt.plot()
