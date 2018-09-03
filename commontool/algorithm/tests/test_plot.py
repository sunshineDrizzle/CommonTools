import numpy as np

from matplotlib import pyplot as plt
from commontool.algorithm.plot import show_bar_value, auto_bar_width


def test_bar_plot():
    x = np.arange(3)
    y = [1, 2, 3]
    x_ticks = ['11', '22', '33']
    width = auto_bar_width(x)
    plt.figure()
    rects = plt.bar(x, y, width, color='r')
    show_bar_value(rects)
    plt.xticks(x, x_ticks)

    x = np.arange(1)
    y = [1.3241]
    width = auto_bar_width(x)
    plt.figure()
    rects = plt.bar(x, y, width, color='g')
    show_bar_value(rects, '.3f')
