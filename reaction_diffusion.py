import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import scipy.signal as sp
from matplotlib.animation import FFMpegWriter
import os
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# diffusion rates
dU = 0.2
dV = 0.1

# default feed kill rate
#f = 0.055
#k = 0.062

# worms
# f = 0.025
# k = 0.056

# "mitosis" simulation
# f = 0.0367
# k = 0.0649

# "coral growth" simulation
# f = 0.0545
# k = 0.062

# spagetti
# k = 0.06264
# f = 0.06100

# big patches
# k = 0.05207
# f = 0.10950

# swirls
k = 0.05204
f = 0.01887

# User inputs
initial_V = 0.5
time_steps = 200
grid_size = 512


# Grid 
U = np.ones((grid_size, grid_size)) 
V = np.full((grid_size, grid_size), initial_V)

# Disturbance
def initialise_v(x):
    if random.random() < 0.05:  
        return 1
    else:
        return 0
initialise_v_vf = np.vectorize(initialise_v)

V = initialise_v_vf(V)

# Diffusion points
kernel = (1/6) * np.array([
    [0.5, 1.0, 0.5],
    [1.0, -6.0, 1.0],
    [0.5, 1.0, 0.5]], dtype='float64')

# Color map
cmap1 = plt.get_cmap('Reds', 128)
cmap2 = plt.get_cmap('YlGn', 128)
newcolors = np.vstack((cmap1(np.linspace(0, 1, 128)),
                       cmap2(np.linspace(0, 1, 128))))
Purples_YlGnBu = ListedColormap(newcolors)

# Plot
fig, ax = plt.subplots()
im = ax.imshow(V, cmap=Purples_YlGnBu, vmin=0, vmax=1)
ax.axis('off')


def init():
    im.set_data(V)
    return [im]

def update(frame):
    global U, V
    U_V_squared = U * V ** 2
    reaction_U = -U_V_squared + f * (1 - U)
    reaction_V = U_V_squared - (k + f) * V
    lap_U = sp.convolve2d(U, kernel, mode='same', boundary='wrap')
    lap_V = sp.convolve2d(V, kernel, mode='same', boundary='wrap') 
    U = dU * lap_U + reaction_U + U
    V = dV * lap_V + reaction_V + V
    im.set_data(V)
    return [im]

ani = matplotlib.animation.FuncAnimation(fig, update, frames = time_steps, init_func=init, blit=True, interval=10)
text = f"Kill rate: {k}\nFeed rate: {f}"
plt.text(550, 300, text, fontsize = 12)
plt.show()
