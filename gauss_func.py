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

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 2D ******************************
# simplification of integrating over real numbers by assuming func is zero beyond +/-10

#func = e^-x^2
f = lambda x : np.exp(-(x**2))
a = 0; b = 1

exact_area = np.sqrt(np.pi)
exact_area_0_1 = 0.746824
columns = ('h', 'trapezoid area', 'error')
approx = []
error = []
errorh = []
arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
arr2 = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
for n in arr :
    h = (b - a) / n
    total_area = 0
    for i in range(0, n) :
        total_area += h * 0.5 * (f(a + i * h) + f(a + (i + 1) * h))
    approx.append(total_area)
    error.append(np.abs(total_area - exact_area_0_1))
    errorh.append(np.abs(total_area - exact_area_0_1) / (h * h))
    #error.append(np.abs(total_area - exact_area))

title_text = 'Error in the Trapezoid Approximation Method, Function: e^-x^2'
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
            [ arr[8], arr2[8], approx[8], error[8], errorh[8]],
            [ arr[9], arr2[9], approx[9], error[9], errorh[9]],
            [ arr[10], arr2[10], approx[10], error[10], errorh[10]],
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

# TRAPEZOID INTEGRATION WITH NON-UNIFORM POINT DISTRIBUTION 2D ******************************



# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 3D ******************************
# simplification of integrating over real numbers by assuming func is zero beyond +/-10

# #func = e^-(x^2 + y^2)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# f = lambda x, y : np.exp(-(x**2 + y**2))
# a = -5 
# b = 5

# N = 32
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

# exact_area = np.pi
# columns = ('h', 'trapezoid area', 'error')
# approx = []
# error = []
# errorh = []
# arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# arr2 = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
# for n in arr :
#     h = (b - a) / n
#     total_area = 0
#     for j in range(0, n) :
#         for i in range(0, n) :
#             total_area += h * h * 0.25 * f(a + i * h, a + j * h)
#             total_area += h * h * 0.25 * f(a + i * h, a + (j + 1) * h)
#             total_area += h * h * 0.25 * f(a + (i + 1) * h, a + j * h)
#             total_area += h * h * 0.25 * f(a + (i + 1) * h, a + (j + 1) * h)
#     approx.append(total_area)
#     error.append(np.abs(total_area - exact_area))
#     errorh.append(np.abs(total_area - exact_area) / (h * h))

# title_text = 'Error in the Trapezoid Approximation Method, Function: e^-(x^2 + y^2)'
# fig_background_color = 'skyblue'
# fig_border = 'steelblue'
# data =  [
#             [ 'h', 'Trapezoid Approximation', 'Error', 'Error/h^2'],
#             [ arr[0], arr2[0], approx[0], error[0], errorh[0]], 
#             [ arr[1], arr2[1], approx[1], error[1], errorh[1]],
#             [ arr[2], arr2[2], approx[2], error[2], errorh[2]],
#             [ arr[3], arr2[3], approx[3], error[3], errorh[3]],
#             [ arr[4], arr2[4], approx[4], error[4], errorh[4]],
#             [ arr[5], arr2[5], approx[5], error[5], errorh[5]],
#             [ arr[6], arr2[6], approx[6], error[6], errorh[6]],
#             [ arr[7], arr2[7], approx[7], error[7], errorh[7]],
#             [ arr[8], arr2[8], approx[8], error[8], errorh[8]],
#             [ arr[9], arr2[9], approx[9], error[9], errorh[9]],
#             [ arr[10], arr2[10], approx[10], error[10], errorh[10]],
#         ]

# # Pop the headers from the data array
# column_headers = data.pop(0)
# row_headers = [x.pop(0) for x in data]
# cell_text = []
# for row in data:
#     cell_text.append([x for x in row])
# # Create the figure. Setting a small pad on tight_layout
# plt.figure(linewidth = 2,
#            edgecolor = fig_border,
#            facecolor = fig_background_color,
#            tight_layout = {'pad':1},
#           )
# # Add a table at the bottom of the axes
# the_table = plt.table(cellText = cell_text,
#                       rowLabels = row_headers,
#                       rowLoc = 'right',
#                       colLabels = column_headers,
#                       loc = 'center')
# # Make the rows taller 
# the_table.scale(1, 1.2)
# the_table.auto_set_font_size(False)
# the_table.set_fontsize(9)
# # Hide axes
# ax = plt.gca()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# # Hide axes border
# plt.box(on=None)
# # Add title
# plt.suptitle(title_text)
# plt.show()