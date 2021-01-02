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

#FIELD LINE STUFF **************************

#electric field at point (x,y) due to the positive charge
def E(Qpos, pos, x, y):
    den = np.hypot(x - pos[0], y - pos[1]) ** 3
    # E = (k * Q / r) * (direction of r)
    return k * Qpos * (x - pos[0]) / den, k * Qpos * (y - pos[1]) / den

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
              density=2, arrowstyle='->', arrowsize=1.5)

# Add filled circles for the charges themselves
charge_colors = {True: '#aa0000', False: '#0000aa'}
for q, pos in charges:
    ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))

ax.set_xlabel('x-position')
ax.set_ylabel('y-position')
ax.set_title('Electric Field Lines')
ax.set_xlim(min_neg, max_pos)
ax.set_ylim(min_neg, max_pos)
ax.set_aspect('equal')
plt.show()
