import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from time import sleep
import emulator_utils


# Set interactive mode
#plt.ion()

def get_cmap(n, name='brg'):
    return plt.cm.get_cmap(name, n)

def plot_map(map):
    n_players = len(np.unique(map)) - 1
    cmap = get_cmap(n_players)
    
    #bounds = np.arange(n_players)
    bounds = [-1.1, -0.9, 2, 3] # Todo
    norm = colors.BoundaryNorm(bounds, n_players)
    
    fig, ax = plt.subplots()
    
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='--', color='k', linewidth=1)
    ax.set_xticks(np.arange(-map.shape[0], map.shape[1], 1));
    ax.set_yticks(np.arange(-map.shape[0], map.shape[1], 1));
    
    #while True:
    #    data = np.random.rand(10, 10) * 20
    ax.imshow(map, cmap=cmap, norm=norm)
    #fig.canvas.draw()
    #plt.pause(1)
    plt.show()
