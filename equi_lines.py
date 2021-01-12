#equipotential lines

import matplotlib.pyplot as plt
import numpy as np

#position of charges
pos = (-1.001, 0)
neg = (1.001, 0)
pos2 = (0, -1.001)
neg2 = (0, 1.001)
#magnitude of charge
Qneg = -1e-9
Qpos = 1e-9
#constant K
k = 9e9
#dimensions
min_neg = -2
max_pos = 2

#electric potential at point (x,y) due to a dipole
def E2(x, y):
    hyppos = np.hypot(x - pos[0], y - pos[1])
    hypneg = np.hypot(x - neg[0], y - neg[1])
    # E_potential = (k * Q / r)
    return k * Qneg / hypneg + k * Qpos / hyppos

#electric potential at point (x,y) due to 4 charges
def E4(x, y):
    hyppos = np.hypot(x - pos[0], y - pos[1])
    hypneg = np.hypot(x - neg[0], y - neg[1])
    hyppos2 = np.hypot(x - pos2[0], y - pos2[1])
    hypneg2 = np.hypot(x - neg2[0], y - neg2[1])
    # E_potential = (k * Q / r)
    return k * Qpos / hypneg + k * Qpos / hyppos + \
            k * Qneg / hypneg2 + k * Qneg / hyppos2 

x = np.linspace(min_neg, max_pos, 1001)
y = np.linspace(min_neg, max_pos, 1001)
X, Y = np.meshgrid(x, y)
Z = E4(X, Y)

l = [1.8 ** x for x in range(-3, 9, 1)]
l = [-x for x in reversed(l)] + [0] + l
plt.contour(X, Y, Z, l, colors='black');
#plt.contour(X, Y, Z, l, cmap='RdGy');
plot = plt.contourf(X, Y, Z, l, cmap='BuPu')
plt.colorbar();
plt.show()