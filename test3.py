from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from mpl_toolkits import mplot3d
from math import e
from sympy.functions import sin, cos, exp


#func = 1-x^2
f = lambda x : 1 - x**2
a = 0; b = 1
exact_area = 2/3

# #func = x^2
# f = lambda x : x**2
# a = 0; b = 1
# exact_area = 1/3

# #func = x^3
# f = lambda x : x**3
# a = 0; b = 1
# exact_area = 1/4

# #func = 2^x
# f = lambda x : 2**x
# a = 0; b = 1
# exact_area = 1/math.log(2)

# #func = 3^x
# f = lambda x : 3**x
# a = 0; b = 1
# exact_area = 2/math.log(3)

# #func = e^x
# f = lambda x : e**x
# a = 0; b = 1
# exact_area = e - 1

# #func = e^2x
# f = lambda x : e**(x * 2)
# a = 0; b = 1
# exact_area = (e*e - 1) / 2

# #func = sin(x)
# f = lambda x : sin(x)
# a = 0; b = 1
# exact_area = 1 - cos(1)

columns = ('h', 'trapezoid area', 'error')
approx = []
error = []
errorh = []
arr = [1, 2, 4, 8, 16, 32, 64, 128]
arr2 = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
for n in arr :
    h = (b - a) / n
    total_area = 0
    for i in range(0, n) :
        total_area += h * 0.5 * (f(a + i * h) + f(a + (i + 1) * h))
    approx.append(total_area)
    error.append(np.abs(total_area - exact_area))
    errorh.append(np.abs(total_area - exact_area) / (h * h))
    

title_text = 'Error in the Trapezoid Approximation Method, Function: 1-x^2'
fig_background_color = 'skyblue'
fig_border = 'steelblue'
data =  [
            [ 'h', 'Trapezoid Approximation', 'Error', 'Error/h^2'],
            [ arr[0], arr2[0], approx[0], error[0], errorh[0]], 
            [ arr[1], arr2[1], approx[1], error[1], errorh[1]],
            [ arr[2], arr2[2], approx[2], error[2], errorh[2]],
            [ arr[3], arr2[3], approx[3], error[3], errorh[3]],
            [ arr[4], arr2[4], approx[4], error[4], errorh[4]],
            [ arr[5], arr2[5], approx[5], error[5], errorh[5]],
            [ arr[6], arr2[6], approx[6], error[6], errorh[6]],
            [ arr[7], arr2[7], approx[7], error[7], errorh[7]],
        ]
# Pop the headers from the data array
column_headers = data.pop(0)
row_headers = [x.pop(0) for x in data]
cell_text = []
for row in data:
    cell_text.append([x for x in row])
# Create the figure. Setting a small pad on tight_layout
plt.figure(linewidth = 2,
           edgecolor = fig_border,
           facecolor = fig_background_color,
           tight_layout = {'pad':1},
          )
# Add a table at the bottom of the axes
the_table = plt.table(cellText = cell_text,
                      rowLabels = row_headers,
                      rowLoc = 'right',
                      colLabels = column_headers,
                      loc = 'center')
# Make the rows taller 
the_table.scale(1, 1.2)
the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
# Hide axes
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Hide axes border
plt.box(on=None)
# Add title
plt.suptitle(title_text)
plt.show()