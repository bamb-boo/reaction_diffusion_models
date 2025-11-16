print("starting")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import scipy.signal
from matplotlib.animation import FFMpegWriter
import os
from matplotlib.colors import ListedColormap


ffmpeg_path = r"C:\ffmpeg\ffmpeg.exe"

def ffmpeg_custom_writer():
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    writer.bin_path = lambda: r"C:\ffmpeg\ffmpeg.exe"  
    return writer


print("saving path found...")

# diffusion rates
dU = 0.2
dV = 0.1

# default feed kill rate
f = 0.055
k = 0.062

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
# k = 0.05204
# f = 0.01887


# convolution kernel
kernel = np.array([
    [0.05, 0.2, 0.05],
    [0.2, -1.0, 0.2],
    [0.05, 0.2, 0.05]
], dtype='float64')

# Size of array
N = 512

A = np.ones((N, N), dtype='float64')
B = np.zeros((N, N), dtype='float64')

print("initializing...")

def initialise_b(x):
    if random.random() < 0.05:  # chance of B=1 is 0.05
        return 1.0
    else:
        return 0.0

# vectorise the initialisation function
initialise_b_vf = np.vectorize(initialise_b)

B = initialise_b_vf(B)

print("animating...")

# Color map
cmap1 = plt.get_cmap('Reds', 128)
cmap2 = plt.get_cmap('YlGn', 128)
newcolors = np.vstack((cmap1(np.linspace(0, 1, 128)),
                       cmap2(np.linspace(0, 1, 128))))
Purples_YlGnBu = ListedColormap(newcolors)


def update(frame):
    global A, B

    AB2 = A * B * B  # element-wise multiplication

    DU = scipy.signal.convolve2d(A, kernel, mode='same', boundary='wrap')
    DV = scipy.signal.convolve2d(B, kernel, mode='same', boundary='wrap')

    A += (dU * DU) - AB2 + (f *  (1 - A))
    B += (dV * DV) + AB2 - ((k + f) * B) 

    im.set_data(B)
    return [im]

fig = plt.figure()
im = plt.imshow(B, interpolation='none', cmap=Purples_YlGnBu, vmin=0, vmax=1)

print("saving animation...")

anim = matplotlib.animation.FuncAnimation(fig, update, frames=3000, interval=50, repeat=False)
text = f"Kill rate: {k}\n\nFeed rate: {f}"
plt.text(518, 300, text, fontsize = 9)

# Save 
anim.save(filename=f"f_{f}__k_{k}.mp4", writer=ffmpeg_custom_writer())

print("done")

