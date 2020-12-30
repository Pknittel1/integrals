import matplotlib.pyplot as plt
import numpy as np
import time as time
from matplotlib.animation import FuncAnimation

neg = (-0.5, 0)
Qneg = -1
pos = (0.5, 0)
Qpos = 1
k = 9e9

rx = 0.4
ry = 0
mass = 1
Q = 0.5
velx = 0
vely = 30000
dt = 0.0000005

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'bo')

def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return ln,

def update(frame):
    global rx, ry, velx, vely
    #x distances
    xneg = rx - neg[0]
    xpos = rx - pos[0]
    #y distances
    yneg = ry - neg[1]
    ypos = ry - pos[1]
    #hypotenuse squared
    hypneg2 = ((xneg ** 2) + (yneg ** 2))
    hyppos2 = ((xpos ** 2) + (ypos ** 2))
    #forces on the charge by the two charges
    Fneg = k * Qneg * Q / (hypneg2)
    Fpos = k * Qpos * Q / (hyppos2)
    #components of net force on the charge 
    Fx = Fneg * xneg / (np.sqrt(hypneg2)) #+ Fpos * np.cos(xpos / (np.sqrt(hyppos2)))
    Fy = Fneg * yneg / (np.sqrt(hypneg2)) #+ Fpos * np.sin(ypos / (np.sqrt(hyppos2)))
    #components of acceleration
    xaccel = Fx / mass
    yaccel = Fy / mass
    #update
    rx = rx + velx * dt
    ry = ry + vely * dt
    velx = velx + xaccel * dt
    vely = vely + yaccel * dt
    #graph
    ln.set_data([rx, rx + Fx/(1e10), neg[0], pos[0]], [ry, ry + Fy/(1e10), neg[1], pos[1]])
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
