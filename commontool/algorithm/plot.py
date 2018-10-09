import numpy as np

from matplotlib import pyplot as plt


def show_bar_value(rects, val_fmt=''):
    """
    show bars' value on the figure automatically
    Reference: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars

    :param rects:
        bars in the matplotlib ax
    :param val_fmt: str
        value format, used to control the value's visualization format
    """
    for rect in rects:
        value = rect.get_height()
        label = '{0:{1}}'.format(value, val_fmt)
        if value < 0:
            plt.text(rect.get_x() + rect.get_width() / 2., value, label, ha='center', va='top')
        else:
            plt.text(rect.get_x()+rect.get_width()/2., value, label, ha='center', va='bottom')


def auto_bar_width(x, item_num=1):
    """
    decide bar width automatically according to the length and interval of x indices.

    :param x: 1-D sequence
        x indices in the matplotlib ax
    :param item_num: integer
        the number of items for plots

    :return width: float
        bar width
    """
    length = len(x)
    bar_num = length * item_num
    if length > 1:
        interval = x[1] - x[0]
        width = (length - 1.0) * interval / bar_num
    else:
        width = 0.1

    return width


class VlineMover(object):
    """
    Move the vertical line when left button is clicked.
    """

    def __init__(self, vline, x_round=False):
        """
        :param vline: Matplotlib Line2D
            the vertical line object
        :param x_round: bool
            If true, round the x index.
        """
        self.vline = vline
        self.x_round = x_round
        self.ax = vline.axes
        self.x = vline.get_xdata()
        self.y = vline.get_ydata()
        self.cid = vline.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.button == 1 and event.inaxes is self.ax:
            if self.x_round:
                self.x = [round(event.xdata)] * 2
            else:
                self.x = [event.xdata] * 2
            self.vline.set_data(self.x, self.y)
            self.vline.figure.canvas.draw()


class VlineMoverPlotter(object):
    """
    plot a figure with vertical line interaction
    """

    def __init__(self, nrows=1, ncols=1, sharex=False, sharey=False,
                 squeese=True, subplot_kw=None, gridspec_kw=None, **fig_kw):

        self.figure, self.axes = plt.subplots(nrows, ncols, sharex, sharey,
                                              squeese, subplot_kw, gridspec_kw, **fig_kw)
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        self.figure.canvas.mpl_connect('button_press_event', self._on_clicked)

        self.axes_twin = np.zeros_like(self.axes, dtype=np.object)
        self.vline_movers = np.zeros_like(self.axes)

    def add_twinx(self, idx=0, r_idx=0, c_idx=0):
        """
        create and save twin axis for self.axes[idx] or self.axes[r_idx, c_idx]

        :param idx: integer
            The index of the self.axes. (Only used when self.axes.ndim == 1)
        :param r_idx: integer
            The row index of the self.axes. (Only used when self.axes.ndim == 2)
        :param c_idx: integer
            The column index of the self.axes. (Only used when self.axes.ndim == 2)
        :return:
        """
        if self.axes.ndim == 1:
            self.axes_twin[idx] = self.axes[idx].twinx()
        elif self.axes.ndim == 2:
            self.axes_twin[r_idx, c_idx] = self.axes[r_idx, c_idx].twinx()

    def add_vline_mover(self, idx=0, r_idx=0, c_idx=0, vline_idx=0, x_round=False):
        """
        add vline mover for each ax

        :param idx: integer
            The index of the self.axes. (Only used when self.axes.ndim == 1)
        :param r_idx: integer
            The row index of the self.axes. (Only used when self.axes.ndim == 2)
        :param c_idx: integer
            The column index of the self.axes. (Only used when self.axes.ndim == 2)
        :param vline_idx: integer
            A index used to initialize the vertical line's position
        :param x_round: bool
            If true, round the x index.
        :return:
        """
        if self.axes.ndim == 1:
            if self.axes_twin[idx] is 0:
                self.vline_movers[idx] = VlineMover(self.axes[idx].axvline(vline_idx), x_round)
            else:
                self.vline_movers[idx] = VlineMover(self.axes_twin[idx].axvline(vline_idx), x_round)
        elif self.axes.ndim == 2:
            if self.axes_twin[r_idx, c_idx] is 0:
                self.vline_movers[r_idx, c_idx] = VlineMover(self.axes[r_idx, c_idx].axvline(vline_idx), x_round)
            else:
                self.vline_movers[r_idx, c_idx] = VlineMover(self.axes_twin[r_idx, c_idx].axvline(vline_idx), x_round)

    def _on_clicked(self, event):
        pass
