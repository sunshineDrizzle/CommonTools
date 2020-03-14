import numpy as np

from scipy.stats.stats import pearsonr
from commontool.ni import algo as ni_algo


def test_intra_connect():
    # prepare data
    maps = np.random.rand(4, 5, 10)
    seed_mask = np.random.randint(0, 3, (4, 5))
    seed_labels = np.unique(seed_mask)
    seed_labels = seed_labels[seed_labels != 0]
    n_seed = len(seed_labels)
    trg_mask = np.random.randint(3, 5, (4, 5))
    trg_labels = np.unique(trg_mask)
    trg_labels = trg_labels[trg_labels != 0]
    n_trg = len(trg_labels)

    # test correlation
    # ground truth
    connections = np.zeros((n_seed, n_trg))
    for i, seed_lbl in enumerate(seed_labels):
        seed_series = np.mean(maps[seed_mask == seed_lbl], 0)
        for j, trg_lbl in enumerate(trg_labels):
            trg_series = np.mean(maps[trg_mask == trg_lbl], 0)
            connections[i, j] = pearsonr(seed_series, trg_series)[0]
    # test
    connection_dict = ni_algo.intra_connect(maps, seed_mask, trg_mask)
    np.testing.assert_equal(seed_labels, connection_dict['seed_label'])
    np.testing.assert_equal(trg_labels, connection_dict['trg_label'])
    np.testing.assert_almost_equal(connections, connection_dict['connection'], 10)

    # test euclidean
    # ground truth
    connections = np.zeros((n_seed, n_trg))
    for i, seed_lbl in enumerate(seed_labels):
        seed_series = np.mean(maps[seed_mask == seed_lbl], 0)
        for j, trg_lbl in enumerate(trg_labels):
            trg_series = np.mean(maps[trg_mask == trg_lbl], 0)
            connections[i, j] = np.linalg.norm(seed_series-trg_series, ord=2)
    # test
    connection_dict = ni_algo.intra_connect(maps, seed_mask, trg_mask, 'euclidean')
    np.testing.assert_equal(seed_labels, connection_dict['seed_label'])
    np.testing.assert_equal(trg_labels, connection_dict['trg_label'])
    np.testing.assert_almost_equal(connections, connection_dict['connection'], 10)
