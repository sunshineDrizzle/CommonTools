import numpy as np

from matplotlib import pyplot as plt


def test_calc_overlap():
    from commontool.algorithm.tool import calc_overlap

    c1 = [1, 2, 3, 4]
    c2 = (3, 4, 5, 6)
    print(calc_overlap(c1, c2))

    array1 = np.array([1, 2, 3, 2])
    array2 = np.array([4, 5, 6, 4])
    print(calc_overlap(array1, array2, 2, 4))


def test_box_sampling():
    from commontool.algorithm.tool import uniform_box_sampling
    from matplotlib.patches import Rectangle

    n_sample = 40
    bounding_box = np.array([[2, 2], [7, 6]])
    dists = bounding_box[1] - bounding_box[0]
    samples = uniform_box_sampling(n_sample, bounding_box)
    rect = Rectangle(bounding_box[0], dists[0], dists[1], edgecolor='r', facecolor='none')

    fig, ax = plt.subplots()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.scatter(samples[:, 0], samples[:, 1])
    ax.add_patch(rect)
    fig.tight_layout()
    plt.show()
