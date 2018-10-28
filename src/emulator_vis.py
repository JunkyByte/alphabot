import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from time import sleep
import emulator_utils
from matplotlib import cm
from copy import copy
from PIL import Image

def plot_map(map):
    m = copy(map)
    n_players = len(np.unique(m)) - 1

    fig, ax = plt.subplots()
    p = np.where(m != -1)
    m[p] += 1
    m = m / np.max(m)
    m = m * 255

    img = Image.fromarray(m)
    img = img.resize((480,480))
    m = np.array(img)
    plt.imshow(m)
    plt.show()
