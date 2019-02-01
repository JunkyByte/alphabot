import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from matplotlib import cm
from PIL import Image


def plot_map(mapp):
    m = mapp
    fig, ax = plt.subplots()
    p = np.where(m != -1)
    m[p] += 1
    m = m / np.max(m)
    m = m * 255

    img = Image.fromarray(m)
    img = img.resize((480, 480))
    m = np.array(img)
    plt.imshow(m)
    plt.show()
