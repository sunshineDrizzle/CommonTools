import numpy as np

from scipy.spatial.distance import cdist


def intra_connect(maps, seed_mask, trg_mask, metric='correlation'):
    """
    Calculate connections between seeds and targets intra the maps.
    Such as functional connectivity and Covariant connection.

    Parameters:
    ----------
    maps: ndarray
        The maps array has at least two dimensions.
        The last dimension will be regarded as series used to calculate connections.
        The dimension number means the number of maps.
    seed_mask: ndarray
        Its shape is same as each map.
        All non-zero values will be regarded as seed labels.
    trg_mask: ndarray
        Its shape is same as each map.
        All non-zero values will be regarded as target labels.
    metric: str
        connection metric
        choices=('correlation', 'euclidean')

    Returns:
    -------
    connection_dict:
        connection: numpy array
            Its shape is (n_seed, n_target).
            The element located in [i, j] is the connection of seed i and target j.
        seed_label: 1D array
            Its shape is (n_seed,).
            Its elements are seed labels corresponding to the rows of connections.
        trg_label: 1D array
            Its shape is (n_target,)
            Its elements are target labels corresponding to the columns of connections.
    """
    # check data
    assert maps.ndim > 1, 'The maps array has at least two dimensions.'
    if seed_mask.shape != maps.shape[:-1] or trg_mask.shape != maps.shape[:-1]:
        raise ValueError("The shape of a mask must be same as each map.")
    metric_choices = ('correlation', 'euclidean')
    assert metric in metric_choices, 'Method {} is not in {}.'.format(metric, metric_choices)

    # prepare seeds
    seed_labels = np.unique(seed_mask)
    seed_labels = seed_labels[seed_labels != 0]
    seed_series = [np.mean(maps[seed_mask == lbl], 0) for lbl in seed_labels]
    seed_series = np.asarray(seed_series)

    # prepare targets
    trg_labels = np.unique(trg_mask)
    trg_labels = trg_labels[trg_labels != 0]
    trg_series = [np.mean(maps[trg_mask == lbl], 0) for lbl in trg_labels]
    trg_series = np.asarray(trg_series)

    # connect
    if metric == 'correlation':
        connections = 1 - cdist(seed_series, trg_series, metric)
    else:
        connections = cdist(seed_series, trg_series, metric)

    # output
    connection_dict = {
        'connection': connections,
        'seed_label': seed_labels,
        'trg_label': trg_labels
    }
    return connection_dict
