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

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 2D PARAMETER TESTING *******************************

# #func = e^-x^2
# f = lambda x : np.exp(-(x**2))
# # a1 = -40
# # b1 = 40
# # exact_area = np.sqrt(np.pi)
# #exact_area = 1.7724538509055159
# # this is the exact area form the interval from 0 to 1
# # a1 = 0
# # b1 = 1
# # # changing the number of digits here gives weird results
# # exact_area = 0.74682413
# exact_area = 0.74682413281229
# approx = []
# error = []
# errorh = []
# arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# arr2 = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 1/2048, 1/4096, 1/8192, 1/16384, 1/32768]
# for n in arr :
#     h = (b1 - a1) / n
#     total_area = 0
#     for i in range(0, n) :
#         total_area += h * 0.5 * (f(a1 + i * h) + f(a1 + (i + 1) * h))
#     approx.append(total_area)
#     error.append(np.abs(total_area - exact_area))
#     errorh.append((np.abs(total_area - exact_area)) / (h * h))

# title_text = 'Error in the Trapezoid Method (changing h), Function: e^-x^2'
# fig_background_color = 'skyblue'
# fig_border = 'steelblue'
# data =  [
#         [ 'h', 'Trapezoid Approximation', 'Error', 'Error/h^2'],
#         [ arr[0], arr2[0], approx[0], error[0], errorh[0]], 
#         [ arr[1], arr2[1], approx[1], error[1], errorh[1]],
#         [ arr[2], arr2[2], approx[2], error[2], errorh[2]],
#         [ arr[3], arr2[3], approx[3], error[3], errorh[3]],
#         [ arr[4], arr2[4], approx[4], error[4], errorh[4]],
#         [ arr[5], arr2[5], approx[5], error[5], errorh[5]],
#         [ arr[6], arr2[6], approx[6], error[6], errorh[6]],
#         [ arr[7], arr2[7], approx[7], error[7], errorh[7]],
#         [ arr[8], arr2[8], approx[8], error[8], errorh[8]],
#         [ arr[9], arr2[9], approx[9], error[9], errorh[9]],
#         [ arr[10], arr2[10], approx[10], error[10], errorh[10]],
#         [ arr[11], arr2[11], approx[11], error[11], errorh[11]],
#         [ arr[12], arr2[12], approx[12], error[12], errorh[12]],
#         [ arr[13], arr2[13], approx[13], error[13], errorh[13]],
#         [ arr[14], arr2[14], approx[14], error[14], errorh[14]],
#         [ arr[15], arr2[15], approx[15], error[15], errorh[15]],
#     ]

# #ERROR PLOT
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(arr2, error, 'ro')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel('h')
# ax.set_ylabel('error')
# ax.set_title('h error plot')

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

# #GRAPH OF FUNCTION
# Num = 64
# # x and y values for the trapezoid rule
# x = np.linspace(a1, b1, Num + 1)
# y = f(x)
# # X and Y values for plotting y=f(x)
# X = np.linspace(a1, b1, 100)
# Y = f(X)
# plt.plot(X,Y)
# for i in range(Num):
#     xs = [x[i], x[i], x[i + 1], x[i + 1]]
#     ys = [0, f(x[i]), f(x[i + 1]), 0]
#     plt.fill(xs, ys, 'b', edgecolor = 'b', alpha = 0.2)
# plt.title('Uniform Trapezoid Rule, N = {}'.format(Num))
# plt.show()

# # changing L **********************************************************************************************

# #func = e^-x^2
# f = lambda x : np.exp(-(x**2))
# #this arrary is the different L/2 tried to increase accuracy of integral
# L = [1, 2, 3, 4, 6, 8, 12, 16, 32]
# exact_area = np.sqrt(np.pi)
# approx2 = []
# error2 = []
# errorh2 = []
# n1 = 4096
# for b in L :
#     a = -b
#     h = (b - a) / n1
#     total_area = 0
#     for i in range(0, n1) :
#         total_area += h * 0.5 * (f(a + i * h) + f(a + (i + 1) * h))
#     approx2.append(total_area)
#     error2.append(np.abs(total_area - exact_area))
#     errorh2.append((np.abs(total_area - exact_area)) / (h * h))

# data2 =  [
#         [ 'L', 'Trapezoid Approximation', 'Error', 'Error/h^2'],
#         [ '.', L[0], approx2[0], error2[0], errorh2[0]], 
#         [ '.', L[1], approx2[1], error2[1], errorh2[1]],
#         [ '.', L[2], approx2[2], error2[2], errorh2[2]],
#         [ '.', L[3], approx2[3], error2[3], errorh2[3]],
#         [ '.', L[4], approx2[4], error2[4], errorh2[4]],
#         [ '.', L[5], approx2[5], error2[5], errorh2[5]],
#         [ '.', L[6], approx2[6], error2[6], errorh2[6]],
#         [ '.', L[7], approx2[7], error2[7], errorh2[7]],
#         [ '.', L[8], approx2[8], error2[8], errorh2[8]],
#     ]

# title_text2 = 'Error in the Trapezoid Method (changing L), Function: e^-x^2'
# fig_background_color = 'skyblue'
# fig_border = 'steelblue'

# #ERROR PLOT
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(L, error2, 'ro')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel('h')
# ax.set_ylabel('error')
# ax.set_title('L error plot')

# # Pop the headers from the data array
# column_headers = data2.pop(0)
# row_headers = [x.pop(0) for x in data2]
# cell_text = []
# for row in data2:
#     cell_text.append([x for x in row])
# # Create the figure. Setting a small pad on tight_layout
# plt.figure(linewidth = 2,
#            edgecolor = fig_border,
#            facecolor = fig_background_color,
#            tight_layout = {'pad':1},
#           )
# # Add a table at the bottom of the axes
# the_table2 = plt.table(cellText = cell_text,
#                       rowLabels = row_headers,
#                       rowLoc = 'right',
#                       colLabels = column_headers,
#                       loc = 'center')
# # Make the rows taller 
# the_table2.scale(1, 1.2)
# the_table2.auto_set_font_size(False)
# the_table2.set_fontsize(9)
# # Hide axes
# ax = plt.gca()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# # Hide axes border
# plt.box(on=None)
# # Add title
# plt.suptitle(title_text2)
# plt.show()

# #GRAPH OF FUNCTION
# Num = 64
# a = -4
# b = 4
# # x and y values for the trapezoid rule
# x = np.linspace(a, b, Num + 1)
# y = f(x)
# # X and Y values for plotting y=f(x)
# X = np.linspace(a, b, 100)
# Y = f(X)
# plt.plot(X,Y)
# for i in range(Num):
#     xs = [x[i], x[i], x[i + 1], x[i + 1]]
#     ys = [0, f(x[i]), f(x[i + 1]), 0]
#     plt.fill(xs, ys, 'b', edgecolor = 'b', alpha = 0.2)
# plt.title('Uniform Trapezoid Rule, N = {}'.format(Num))
# plt.show()

# TRAPEZOID INTEGRATION 2D PARAMETER TESTING comparing error of h and L *************************************

#func = e^-x^2
f = lambda x : np.exp(-(x**2))
#this arrary is the different L/2 tried to increase accuracy of integral
L = [1, 2, 4, 8, 16]
labL = ['L = 1', 'L = 2', 'L = 4', 'L = 8', 'L = 16']
# exact_area = np.sqrt(np.pi)
exact_area = 0.74682413281229
columns = ('h', 'trapezoid area', 'error') 
arr = [1, 2, 4, 8, 16, 32, 64, 128]
labh = [ 'h = 1', 'h = 1/2', 'h = 1/4', 'h = 1/8', 'h = 1/16', 'h = 1/32', 'h = 1/64', 'h = 1/128']
# Creates a list containing arr lists, each of L items, all set to -1
e = [[-1 for x in range(len(L))] for y in range(len(arr))] 
i = 0
for n in arr :
    j = 0
    for b in L:
        a = 0
        h = (b - a) / n
        total_area = 0
        for k in range(0, n) :
            total_area += h * 0.5 * (f(a + k * h) + f(a + (k + 1) * h))
        e[i][j] = np.abs(total_area - exact_area)
        j += 1
    i += 1

title_text = 'Comparing Error in Changing L and h, Function: e^-x^2'
fig_background_color = 'skyblue'
fig_border = 'steelblue'
data =  [
        [ labL[0], labL[1], labL[2], labL[3], labL[4]],
        [ labh[0], e[0][0], e[0][1], e[0][2], e[0][3]],  
        [ labh[1], e[1][0], e[1][1], e[1][2], e[1][3]],
        [ labh[2], e[2][0], e[2][1], e[2][2], e[2][3]],
        [ labh[3], e[3][0], e[3][1], e[3][2], e[3][3]],
        [ labh[4], e[4][0], e[4][1], e[4][2], e[4][3]],
        [ labh[5], e[5][0], e[5][1], e[5][2], e[5][3]],
        [ labh[6], e[6][0], e[6][1], e[6][2], e[6][3]],
        [ labh[7], e[7][0], e[7][1], e[7][2], e[7][3]],
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
the_table.set_fontsize(8)
# Hide axes
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Hide axes border
plt.box(on=None)
# Add title
plt.suptitle(title_text)
plt.show()

# #GRAPH OF FUNCTION
# N = 4
# # x and y values for the trapezoid rule
# x = np.linspace(a, b, N+1)
# y = f(x)
# # X and Y values for plotting y=f(x)
# X = np.linspace(a, b, 100)
# Y = f(X)
# plt.plot(X,Y)
# for i in range(N):
#     xs = [x[i], x[i], x[i+1], x[i+1]]
#     ys = [0, f(x[i]), f(x[i+1]), 0]
#     plt.fill(xs, ys, 'b', edgecolor = 'b', alpha = 0.2)
# plt.title('Uniform Trapezoid Rule, N = {}'.format(N))
# plt.show()

# TRAPEZOID INTEGRATION WITH UNIFORM POINT DISTRIBUTION 3D **************************************************
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