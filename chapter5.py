# #Chapter 5 reproduction of material from Prof Krasny's notes Math 471
# #Interpolation

from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np


### ****************** TAYLOR SERIES EXPANSION *********************

# plt.rcParams['lines.linewidth'] = 2

# # Define symbol
# x = Symbol('x')

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
# p = plot(func, taylor(func,0,1), taylor(func,0,3), taylor(func,0,5),
#          (x,-0.2,0.2),legend=True, show=False)

# p[0].line_color = 'blue'
# p[1].line_color = 'green'
# p[2].line_color = 'firebrick'
# p[3].line_color = 'cyan'
# p.title = 'Taylor Series Expansion'
# p.show()

### ************** END OF TAYLOR SERIES EXPANSION *****************

### ****************** LAGRANGE INTERPOLATION *********************

