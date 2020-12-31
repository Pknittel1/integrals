import matplotlib.pyplot as plt
import numpy as np
import time as time
from matplotlib.animation import FuncAnimation

neg = (-2, 0)
Qneg = -2
pos = (2, 0)
Qpos = 3
k = 9e9
dt = 0.0000005
t = 0

#particle 1
mass = 1
Q = 5
rx = 0.34
ry = 2.6
velx = 10
vely = 50
xaccel = 0
yaccel = 0

#Graphs
fig, ax = plt.subplots(3, 3)
#make extra space between subplots
fig.subplots_adjust(hspace = 0.5)
ln1, = ax[0,0].plot([], [], 'bo')
ax[0,0].set_title('Motion of particle')
ax[0,0].set_xlabel('x-position')
ax[0,0].set_ylabel('y-position')
ax[0,0].grid(True)
#position versus time plots
ln2, = ax[0,1].plot([], [], 'go')
ax[0,1].set_title('Particle x position vs. time')
ax[0,1].set_xlabel('time')
ax[0,1].set_ylabel('x-position')
ax[0,1].grid(True)
ln3, = ax[0,2].plot([], [], 'go')
ax[0,2].set_title('Particle y position vs. time')
ax[0,2].set_xlabel('time')
ax[0,2].set_ylabel('y-position')
ax[0,2].grid(True)
#velocity versus time plots
ln4, = ax[1,0].plot([], [], 'ro')
ax[1,0].set_title('Particle x velocity vs. time')
ax[1,0].set_xlabel('time')
ax[1,0].set_ylabel('x-velocity')
ax[1,0].grid(True)
ln5, = ax[1,1].plot([], [], 'ro')
ax[1,1].set_title('Particle y velocity vs. time')
ax[1,1].set_xlabel('time')
ax[1,1].set_ylabel('y-velocity')
ax[1,1].grid(True)
#acceleration versus time plots
ln6, = ax[1,2].plot([], [], 'co')
ax[1,2].set_title('Particle x acceleration vs. time')
ax[1,2].set_xlabel('time')
ax[1,2].set_ylabel('x-acceleration')
ax[1,2].grid(True)
ln7, = ax[2,0].plot([], [], 'co')
ax[2,0].set_title('Particle y acceleration vs. time')
ax[2,0].set_xlabel('time')
ax[2,0].set_ylabel('y-acceleration')
ax[2,0].grid(True)

def init():
    ax[0,0].set_xlim(-10, 10)
    ax[0,0].set_ylim(-10, 10)
    ax[0,1].set_xlim(-1, 10)
    ax[0,1].set_ylim(-10, 10)
    ax[0,2].set_xlim(-1, 10)
    ax[0,2].set_ylim(-10, 10)
    ax[1,0].set_xlim(-1, 10)
    ax[1,0].set_ylim(-100, 100)
    ax[1,1].set_xlim(-1, 10)
    ax[1,1].set_ylim(-100, 100)
    ax[1,2].set_xlim(-1, 10)
    ax[1,2].set_ylim(-100, 100)
    ax[2,0].set_xlim(-1, 10)
    ax[2,0].set_ylim(-100, 100)
    ax[2,1].set_xlim(-5, 5)
    ax[2,1].set_ylim(-5, 5)
    ax[2,2].set_xlim(-5, 5)
    ax[2,2].set_ylim(-5, 5)
    return ln1, ln2, ln3, ln4, ln5, ln6, ln7

def update(frame):
    global rx, ry, velx, vely, t, xaccel, yaccel
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
    Fx = Fneg * xneg / (np.sqrt(hypneg2)) + Fpos * xpos / (np.sqrt(hyppos2))
    Fy = Fneg * yneg / (np.sqrt(hypneg2)) + Fpos * ypos / (np.sqrt(hyppos2))
    #components of acceleration
    xaccel = Fx / mass
    yaccel = Fy / mass
    #update
    t = t + dt * 100000
    rx = rx + velx * dt
    ry = ry + vely * dt
    velx = velx + xaccel * dt
    vely = vely + yaccel * dt
    #graph
    #ln.set_data([rx, rx + Fx/(1e10), neg[0], pos[0]], [ry, ry + Fy/(1e10), neg[1], pos[1]])
    ln1.set_data([rx, neg[0], pos[0]], [ry, neg[1], pos[1]])
    ln2.set_data([t,], [rx,])
    ln3.set_data([t,], [ry,])
    ln4.set_data([t,], [velx  / 1e4,])
    ln5.set_data([t,], [vely  / 1e4,])
    ln6.set_data([t,], [xaccel / 1e8,])
    ln7.set_data([t,], [yaccel / 1e8,])
    return ln1, ln2, ln3, ln4, ln5, ln6, ln7

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
