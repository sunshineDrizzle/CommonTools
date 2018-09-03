from matplotlib import pyplot as plt


def show_bar_value(rects):
    """
    show bars' value on the figure automatically
    Reference: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars

    :param rects:
        bars in the matplotlib ax
    """
    for rect in rects:
        h = rect.get_height()
        if h < 0:
            plt.text(rect.get_x() + rect.get_width() / 2., h, str(h), ha='center', va='top')
        else:
            plt.text(rect.get_x()+rect.get_width()/2., h, str(h), ha='center', va='bottom')


def auto_bar_width(x):
    """
    decide bar width automatically according to the length and interval of x indices.

    :param x: 1-D sequence
        x indices in the matplotlib ax

    :return width: float
        bar width
    """
    length = len(x)
    if length > 1:
        interval = x[1] - x[0]
        width = (length - 1) * interval / length
    else:
        width = 0.1

    return width
