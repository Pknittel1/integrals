#equipotential lines

import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Circle

#starting position on circle of charges
pos = (-1, 0)
#magnitude of charge
Q = 1e-9
#constant K
k = 9e9
#number of charge pairs 
num_char = 1
#number of places that E-feild is being calculated at
num_q = 64 * 64
#dimensions
min_neg = -2
max_pos = 2

#electric potential at point (x,y) due to the positive charge
def E(x, y):
    hyp = np.hypot(x - pos[0], y - pos[1])
    # E_potential = (k * Q / r)
    return k * Q / hyp

#grid of x * y points
nx, ny = 64, 64
x = np.linspace(min_neg, max_pos, nx)
y = np.linspace(min_neg, max_pos, ny)
X, Y = np.meshgrid(x, y)
Z = E(X, Y)

#multipole with 2 * num_char charges of alternating sign, equally spaced
#on the unit circle
num_q = 2 ** int(num_char)
charges = []
for i in range(num_q):
    Q = (i % 2) * 2 - 1
    charges.append((Q, (np.cos(2 * np.pi * i / num_q), np.sin(2 * np.pi * i / num_q))))

l = [1.8 ** x for x in range(-3, 9, 1)]
l = [-x for x in reversed(l)] + [0] + l

plt.contour(X, Y, Z, l, colors='black');

plot = plt.contourf(X, Y, Z, l, cmap='BuPu')
plt.colorbar();

plt.show()