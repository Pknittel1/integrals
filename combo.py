#Combination of equipotential and field lines of a dipole
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Circle

#position of charges
pos = (-1.001, 0)
neg = (1.001, 0)
pos2 = (0, -1.001)
neg2 = (0, 1.001)
#magnitude of charge
Qneg = -1e-9
Qpos = 1e-9
#magnitude of charge
Q = 1e-9
#constant K
k = 9e9
#dimensions
min_neg = -2
max_pos = 2
#starting position on circle of charges
pos = (-1, 0)
#number of charge pairs 
num_char = 1
#number of places that E-feild is being calculated at
num_q = 64 * 64

#Electric field lines ************************************

#electric field at point (x,y) due to the positive charge
def E(Q, pos, x, y):
    den = np.hypot(x - pos[0], y - pos[1]) ** 3
    # E = (k * Q / r^r) * (direction of r)
    return k * Q * (x - pos[0]) / den, k * Q * (y - pos[1]) / den

#grid of x * y points
nx, ny = 64, 64
x = np.linspace(min_neg, max_pos, nx)
y = np.linspace(min_neg, max_pos, ny)
X, Y = np.meshgrid(x, y)

#multipole with 2 * num_char charges of alternating sign, equally spaced
#on the unit circle
num_q = 2 ** int(num_char)
charges = []
for i in range(num_q):
    Q = (i % 2) * 2 - 1
    charges.append((Q, (np.cos(2 * np.pi * i / num_q), np.sin(2 * np.pi * i / num_q))))

# Electric field vector in components, E = (Ex, Ey)
Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
for charge in charges:
    ex, ey = E(*charge, x = X, y = Y)
    Ex += ex
    Ey += ey
    fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the streamlines with an appropriate colormap and arrow style
color = 2 * np.log(np.hypot(Ex, Ey))
ax.streamplot(x, y, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=1, arrowstyle='->', arrowsize=1.5)

# Add filled circles for the charges themselves
charge_colors = {True: '#aa0000', False: '#0000aa'}
for q, pos in charges:
    ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))

ax.set_xlabel('x-position')
ax.set_ylabel('y-position')
ax.set_title('Electric Field and Equipotential Lines')
ax.set_xlim(min_neg, max_pos)
ax.set_ylim(min_neg, max_pos)
ax.set_aspect('equal')

#equipotential lines ************************************

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
Z = E2(X, Y)
#logarithmic scaling of colors
l = [1.8 ** x for x in range(-3, 9, 1)]
l = [-x for x in reversed(l)] + [0] + l
plt.contour(X, Y, Z, l, colors='black');
plot = plt.contourf(X, Y, Z, l, cmap='BuPu')
plt.colorbar();

plt.show()