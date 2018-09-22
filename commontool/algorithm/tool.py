import numpy as np


def _overlap(c1, c2, index='dice'):
    """
    Calculate overlap between two collections

    Parameters
    ----------
    c1, c2 : collection (list | tuple | set | 1-D array etc.)
    index : string ('dice' | 'percent')
        This parameter is used to specify index which is used to measure overlap.

    Return
    ------
    overlap : float
        The overlap between c1 and c2
    """
    set1 = set(c1)
    set2 = set(c2)
    intersection_num = float(len(set1 & set2))
    try:
        if index == 'dice':
            total_num = len(set1 | set2) + intersection_num
            overlap = 2.0 * intersection_num / total_num
        elif index == 'percent':
            overlap = 1.0 * intersection_num / len(set1)
        else:
            raise Exception("Only support 'dice' and 'percent' as overlap indices at present.")
    except ZeroDivisionError as e:
        print(e)
        overlap = np.nan
    return overlap


def calc_overlap(data1, data2, label1=None, label2=None, index='dice'):
    """
    Calculate overlap between two sets.
    The sets are acquired from data1 and data2 respectively.

    Parameters
    ----------
    data1, data2 : collection or numpy array
        label1 is corresponding with data1
        label2 is corresponding with data2
    label1, label2 : None or labels
        If label1 or label2 is None, the corresponding data is supposed to be
        a collection of members such as vertices and voxels.
        If label1 or label2 is a label, the corresponding data is always a numpy array with same shape and meaning.
        And we will acquire set1 elements whose labels are equal to label1 from data1
        and set2 elements whose labels are equal to label2 from data2.
    index : string ('dice' | 'percent')
        This parameter is used to specify index which is used to measure overlap.

    Return
    ------
    overlap : float
        The overlap of data1 and data2
    """
    if label1 is not None:
        positions1 = np.where(data1 == label1)
        data1 = list(zip(*positions1))

    if label2 is not None:
        positions2 = np.where(data2 == label2)
        data2 = list(zip(*positions2))

    # calculate overlap
    overlap = _overlap(data1, data2, index)

    return overlap


# --------------matplotlib plot tools--------------
class VlineMover(object):
    """
    Move the vertical line when left button is clicked.
    """

    def __init__(self, vline, x_round=False):
        """
        :param vline: Matplotlib Line2D
            the vertical line object
        :param x_round: bool
            If true, round the x index
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
