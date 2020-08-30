import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from commontool.vis import plot as ct_plot


def test_polyfit_plot():
    # observation by naked eye

    # prepare data
    x = np.arange(-1, 1, 0.02)
    y1 = 2 * np.sin(x * 2.3) + np.random.rand(len(x))
    y2 = 2 * np.sin(x * 2.3) + np.random.rand(len(x)) + 2
    x_min, x_max = np.min(x), np.max(x)
    x_plot = np.linspace(x_min, x_max, 100)

    # test degree=3
    y_plot1_np = np.poly1d(np.polyfit(x, y1, 3))(x_plot)
    y_plot2_np = np.poly1d(np.polyfit(x, y2, 3))(x_plot)
    plt.figure('numpy')
    plt.scatter(x, y1)
    plt.plot(x_plot, y_plot1_np)
    plt.scatter(x, y2)
    plt.plot(x_plot, y_plot2_np)
    plt.legend(['1', '2'])

    plt.figure('commontool')
    ct_plot.polyfit_plot(x, y1, 3)
    ct_plot.polyfit_plot(x, y2, 3)
    plt.legend(['1', '2'])

    # test degree=1
    y_plot1_glm = LinearRegression().fit(x[:, None], y1).predict(x_plot[:, None])
    y_plot2_glm = LinearRegression().fit(x[:, None], y2).predict(x_plot[:, None])
    plt.figure('glm')
    plt.scatter(x, y1)
    plt.plot(x_plot, y_plot1_glm)
    plt.scatter(x, y2)
    plt.plot(x_plot, y_plot2_glm)
    plt.legend(['1', '2'])

    plt.figure('commontool_1d')
    ct_plot.polyfit_plot(x, y1, 1)
    ct_plot.polyfit_plot(x, y2, 1)
    plt.legend(['1', '2'])

    plt.show()
