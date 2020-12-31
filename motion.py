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
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, qx8, ax9)) = plt.subplots(3, 3)
#make extra space between subplots
fig.subplots_adjust(hspace = 0.5)
ln1, = ax1.plot([], [], 'bo')
ax1.set_title('Motion of particle')
ax1.set_xlabel('x-position')
ax1.set_ylabel('y-position')
ax1.grid(True)
#position versus time plots
ln2, = ax2.plot([], [], 'go')
ax2.set_title('Particle x position vs. time')
ax2.set_xlabel('x-position')
ax2.set_ylabel('time')
ax2.grid(True)
ln3, = ax3.plot([], [], 'go')
ax3.set_title('Particle y position vs. time')
ax3.set_xlabel('y-position')
ax3.set_ylabel('time')
ax3.grid(True)
#velocity versus time plots
ln4, = ax4.plot([], [], 'go')
ax4.set_title('Particle x velocity vs. time')
ax4.set_xlabel('x-velocity')
ax4.set_ylabel('time')
ax4.grid(True)
ln5, = ax5.plot([], [], 'ro')
ax5.set_title('Particle y velocity vs. time')
ax5.set_xlabel('y-velocity')
ax5.set_ylabel('time')
ax5.grid(True)
#acceleration versus time plots
ln6, = ax6.plot([], [], 'co')
ax6.set_title('Particle x acceleration vs. time')
ax6.set_xlabel('x-acceleration')
ax6.set_ylabel('time')
ax6.grid(True)
ln7, = ax7.plot([], [], 'co')
ax7.set_title('Particle y acceleration vs. time')
ax7.set_xlabel('y-acceleration')
ax7.set_ylabel('time')
ax7.grid(True)

def init():
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax4.set_xlim(-50, 50)
    ax4.set_ylim(-50, 50)
    ax5.set_xlim(-50, 50)
    ax5.set_ylim(-50, 50)
    ax6.set_xlim(-1000, 1000)
    ax6.set_ylim(-1000, 1000)
    ax7.set_xlim(-100, 100)
    ax7.set_ylim(-100, 100)
    ax8.set_xlim(-5, 5)
    ax8.set_ylim(-5, 5)
    ax9.set_xlim(-5, 5)
    ax9.set_ylim(-5, 5)
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
    t = t + dt
    rx = rx + velx * dt
    ry = ry + vely * dt
    velx = velx + xaccel * dt
    vely = vely + yaccel * dt
    #graph
    #ln.set_data([rx, rx + Fx/(1e10), neg[0], pos[0]], [ry, ry + Fy/(1e10), neg[1], pos[1]])
    ln1.set_data([rx, neg[0], pos[0]], [ry, neg[1], pos[1]])
    ln2.set_data([rx,], [t,])
    ln3.set_data([ry,], [t,])
    ln4.set_data([velx,], [t,])
    ln5.set_data([vely,], [t,])
    ln6.set_data([xaccel,], [t,])
    ln7.set_data([yaccel,], [t,])
    return ln1, ln2, ln3, ln4, ln5, ln6, ln7

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
