import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from time import sleep
import emulator_utils

## TODO

# Define MAP
PLAYERS = 2
MAP_SIZE = emulator_utils.get_map_size(PLAYERS)
map = np.empty(())
data = np.random.rand(10, 10) * 20

plt.ion()

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(10)
bounds = [0, 1, 2, 10, 20]
norm = colors.BoundaryNorm(bounds, 10)

fig, ax = plt.subplots()

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, 10, 1));
ax.set_yticks(np.arange(-.5, 10, 1));

while True:
    data = np.random.rand(10, 10) * 20
    ax.imshow(data, cmap=cmap, norm=norm)
    fig.canvas.draw()
    plt.pause(1)
