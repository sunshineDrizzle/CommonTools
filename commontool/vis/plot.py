import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


def polyfit_plot(x, y, deg, scoring='r'):
    """

    Parameters
    ----------
    x : ndarray, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : ndarray, shape (M,)
        y-coordinates of the M sample points ``(x[i], y[i])``.
    deg : int
        Degree of the fitting polynomial
    scoring : str
        a method used to evaluate the fit effect

    References
    ----------
    1. https://blog.csdn.net/fffsolomon/article/details/104831050
    """

    # fit and construct polynomial
    coefs = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefs)
    print('polynomial:\n', polynomial)

    # scoring
    y_pred = polynomial(x)
    if scoring == 'r':
        score = pearsonr(y, y_pred)[0]
    elif scoring == 'r2_score':
        score = r2_score(y, y_pred)
    else:
        raise ValueError("Not supported scoring:", scoring)
    print('\nscore:', score)

    # plot scatter
    plt.scatter(x, y)

    # plot fitted curve
    x_min, x_max = np.min(x), np.max(x)
    x_plot = np.linspace(x_min, x_max, 100)
    y_plot = polynomial(x_plot)
    plt.plot(x_plot, y_plot)
