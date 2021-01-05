import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

#position of charges
pos = (-1.001, 0)
neg = (1.001, 0)
#magnitude of charge
Qneg = -1e-9
Qpos = 1e-9
#constant K
k = 9e9
#dimensions
min_neg = -2
max_pos = 2

#electric potential at point (x,y) due to the positive charge
def E(x, y):
    hyppos = np.hypot(x - pos[0], y - pos[1])
    hypneg = np.hypot(x - neg[0], y - neg[1])
    # E_potential = (k * Q / r)
    return k * Qneg / hypneg + k * Qpos / hyppos

x = np.linspace(min_neg, max_pos, 1001)
y = np.linspace(min_neg, max_pos, 1001)

X, Y = np.meshgrid(x, y)
Z = E(X, Y)

l = [1.8 ** x for x in range(-3, 9, 1)]
l = [-x for x in reversed(l)] + [0] + l

plt.contour(X, Y, Z, l, colors='black');

#plt.contour(X, Y, Z, l, cmap='RdGy');

plt.contourf(X, Y, Z, l, cmap='BuPu')
plt.colorbar();

#ax.set_xlabel('x-position')
#ax.set_ylabel('y-position')
#ax.set_title('Equipotential Lines')

plt.show()