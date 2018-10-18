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


def test_gap_statistic():
    from commontool.algorithm.tool import gap_statistic, uniform_box_sampling
    from matplotlib.patches import Rectangle

    cluster1_x = np.random.normal(2, 0.2, 100)
    cluster1_y = np.random.normal(1, 0.2, 100)
    cluster1 = np.c_[cluster1_x, cluster1_y]
    cluster2_x = np.random.normal(3, 0.2, 100)
    cluster2_y = np.random.normal(3, 0.2, 100)
    cluster2 = np.c_[cluster2_x, cluster2_y]
    cluster3_x = np.random.normal(5, 0.2, 100)
    cluster3_y = np.random.normal(1, 0.2, 100)
    cluster3 = np.c_[cluster3_x, cluster3_y]
    data = np.r_[cluster1, cluster2, cluster3]

    cluster_nums = range(1, 11)
    labels_list, Wks, Wks_refs_log_mean, gaps, s, k_selected = gap_statistic(data, cluster_nums)

    minimums = np.atleast_2d(np.min(data, 0))
    maximums = np.atleast_2d(np.max(data, 0))
    bounding_box = np.r_[minimums, maximums]
    dists = bounding_box[1] - bounding_box[0]
    rect = Rectangle(bounding_box[0], dists[0], dists[1], edgecolor='r', facecolor='none')
    uniform_samples = uniform_box_sampling(data.shape[0], bounding_box)

    fig1, ax1 = plt.subplots()
    ax1.set_xlim([0, 6])
    ax1.set_ylim([0, 5])
    ax1.scatter(data[:, 0], data[:, 1])
    ax1.scatter(uniform_samples[:, 0], uniform_samples[:, 1], c='r')
    ax1.add_patch(rect)
    fig1.tight_layout()

    x = np.arange(len(cluster_nums))
    plt.figure()
    plt.plot(x, Wks, 'b.-')
    plt.xlabel('#cluster')
    plt.ylabel(('W\u2096'))
    plt.xticks(x, cluster_nums)

    Wks_log = np.log(Wks)
    plt.figure()
    plt.plot(x, Wks_log, 'b.-')
    plt.plot(x, Wks_refs_log_mean, 'r.-')
    plt.xlabel('#cluster')
    plt.ylabel('log(W\u2096)')
    plt.xticks(x, cluster_nums)

    plt.figure()
    plt.plot(x, gaps, 'b.-')
    plt.fill_between(x, gaps-s, gaps+s, alpha=0.5)
    plt.xlabel('#cluster')
    plt.ylabel('gap statistic')
    plt.xticks(x, cluster_nums)

    plt.show()
