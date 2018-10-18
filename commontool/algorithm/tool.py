import numpy as np

from scipy.spatial.distance import cdist, pdist


# --------metrics--------
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


def elbow_score(X, labels, metric='euclidean', type=('inner', 'centroid')):
    """
    calculate elbow score for a partition specified by labels
    https://en.wikipedia.org/wiki/Elbow_method_(clustering)

    :param X: array, shape = (n_samples, n_features)
        a feature array
    :param labels: array, shape = (n_samples,)
        Predicted labels for each sample.
    :param metric: string
        Specify how to calculate distance between samples in a feature array.
        Options: 'euclidean', 'correlation'
    :param type: tuple of two strings
        Options:
        ('inner', 'centroid') - For each cluster, calculate the metric between samples within it
                                with the cluster's centroid. Finally, average all samples.
        ('inner', 'pairwise') - For each cluster, calculate the metric pairwise among samples within it.
                                Finally, average all samples.
        ('inter', 'centroid') - Calculate the metric between cluster centroids with their centroid.
                                Finally, average all clusters.
        ('inter', 'pairwise') - Calculate the metric pairwise among cluster centroids.
                                Finally, average all clusters.

    :return: score:
        elbow score of the partition
    """
    if type == ('inner', 'centroid'):
        # https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
        # formula-1 in (Goutte, Toft et al. 1999 - NeuroImage)
        sub_scores = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_samples_centroid = np.atleast_2d(np.mean(sub_samples, 0))
            tmp_scores = cdist(sub_samples_centroid, sub_samples, metric=metric)[0]
            sub_scores.extend(tmp_scores)
        score = np.mean(sub_scores)
    elif type == ('inner', 'pairwise'):
        sub_scores = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_scores.extend(pdist(sub_samples, metric=metric))
        score = np.mean(sub_scores)
    elif type == ('inter', 'centroid'):
        # adapted from formula-2 in (Goutte, Toft et al. 1999 - NeuroImage)
        sub_centroids = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_centroids.append(np.mean(sub_samples, 0))
        centroid = np.atleast_2d(np.mean(sub_centroids, 0))
        tmp_scores = cdist(centroid, np.array(sub_centroids), metric=metric)[0]
        score = np.mean(tmp_scores)
    elif type == ('inter', 'pairwise'):
        sub_centroids = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_centroids.append(np.mean(sub_samples, 0))
        score = np.mean(pdist(np.array(sub_centroids), metric=metric))
    else:
        raise TypeError('Type-{} is not supported at present.'.format(type))

    return score


# --------sampling--------
def uniform_box_sampling(n_sample, bounding_box=((0,), (1,))):
    """
    create n_sample samples with uniform distribution in the box
    https://blog.csdn.net/baidu_17640849/article/details/70769555
    https://datasciencelab.wordpress.com/tag/gap-statistic/

    :param n_sample: integer
        the number of samples
    :param bounding_box: array-like, shape = (2, n_dim)
        Shape[1] is the number of dimensions.
        Bounding_box[0] are n_dim minimums of their own dimensions.
        Bounding_box[1] are n_dim maximums of their own dimensions.

    :return: samples: array, shape = (n_sample, n_dim)
    """
    bounding_box = np.array(bounding_box)
    dists = np.diag(bounding_box[1] - bounding_box[0])
    samples = np.random.random_sample((n_sample, bounding_box.shape[1]))
    samples = np.matmul(samples, dists) + bounding_box[0]

    return samples
