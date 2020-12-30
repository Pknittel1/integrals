import matplotlib.pyplot as plt
import numpy as np

neg = (-0.5, 0)
Qneg = -1
Qpos = 1
pos = (0.5, 0)
k = 9e9
rx = 0
ry = 0

for count in range(0):
    sqrt1 = np.sqrt(((neg[0] - rx) * (neg[0] - rx)) + ((neg[1] - rx) * (neg[1] - rx)))
    sqrt2 = np.sqrt(((pos[0] - ry) * (pos[0] - ry)) + ((pos[1] - ry) * (pos[1] - ry)))
    Eneg = k * Qneg / (sqrt1)
    Epos = k * Qpos / (sqrt2)
    


#negative charge
plt.plot(*neg, 'ro')
#positive charge
plt.plot(*pos, 'co')

plt.show()
