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


def elbow_score(X, labels, metric='euclidean', type=('inner', 'standard')):
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
        ('inner', 'standard') - Implement Wk in (Tibshirani et al., 2001b)
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
    if type == ('inner', 'standard'):
        score = 0
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            dists = cdist(sub_samples, sub_samples, metric=metric)
            tmp_score = np.sum(dists) / (2.0 * sub_samples.shape[0])
            score += tmp_score
    elif type == ('inner', 'centroid'):
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
        sub_centroids = np.array(sub_centroids)
        if sub_centroids.shape[0] == 1:
            sub_centroids = np.r_[sub_centroids, sub_centroids]
        score = np.mean(pdist(sub_centroids, metric=metric))
    else:
        raise TypeError('Type-{} is not supported at present.'.format(type))

    return score


def gap_statistic(X, cluster_nums, ref_num=10, cluster_method=None):
    """
    do clustering with gap statistic assessment according to (Tibshirani et al., 2001b)
    https://blog.csdn.net/baidu_17640849/article/details/70769555
    https://datasciencelab.wordpress.com/tag/gap-statistic/
    https://github.com/milesgranger/gap_statistic

    :param X: array, shape = (n_samples, n_features)
        a feature array
    :param cluster_nums: a iterator of integers
        Each integer is the number of clusters to try on the data.
    :param ref_num: integer
        The number of random reference data sets used as inertia reference to actual data.
    :param cluster_method: callable
        The cluster method to do clustering on the feature array. And the method returns
        labels_list (cluster results of each cluster_num in cluster_nums).
        If is None, a default K-means method will be used.

    :return: labels_list: list
        cluster results of each cluster_num in cluster_nums
    :return: Wks: array, shape = (len(cluster_nums),)
        within-cluster dispersion of each cluster_num clustering on the feature array X
    :return: Wks_refs_log_mean: array, shape = (len(cluster_nums),)
        mean within-cluster dispersion of each cluster_num clustering on ref_num reference data sets
    :return: gaps: array, shape = (len(cluster_nums),)
        Wks_refs_log_mean - np.log(Wks)
    :return: s: array, shape = (len(cluster_nums),)
        I think elements in s can be regarded as standard errors of gaps.
    :return: k_selected: integer
        cluster k_selected clusters on X may be the best choice
    """
    if cluster_method is None:
        def k_means(data, cluster_nums):
            """
            http://scikit-learn.org/stable/modules/clustering.html#k-means
            """
            from sklearn.cluster import KMeans

            labels_list = []
            for cluster_num in cluster_nums:
                kmeans = KMeans(cluster_num, random_state=0, n_init=10).fit(data)
                labels_list.append(kmeans.labels_ + 1)
                print('KMeans finished: {}'.format(cluster_num))
            return labels_list

        cluster_method = k_means

    print('Start: calculate W\u2096s')
    Wks = []
    labels_list = cluster_method(X, cluster_nums)
    for labels in labels_list:
        Wks.append(elbow_score(X, labels))
    Wks = np.array(Wks)
    Wks_log = np.log(Wks)
    print('Finish: calculate W\u2096s')

    print("Start: calculate references' W\u2096s")
    Wks_refs_log = []
    minimums = np.atleast_2d(np.min(X, axis=0))
    maximums = np.atleast_2d(np.max(X, axis=0))
    bounding_box = np.r_[minimums, maximums]
    for i in range(ref_num):
        X_ref = uniform_box_sampling(X.shape[0], bounding_box)
        labels_list_ref = cluster_method(X_ref, cluster_nums)
        Wks_ref_log = []
        for labels in labels_list_ref:
            Wks_ref_log.append(np.log(elbow_score(X_ref, labels)))
        Wks_refs_log.append(Wks_ref_log)
        print('Finish reference: {}/{}'.format(i+1, ref_num))
    print("Finish: calculate references' W\u2096s")

    print('Start: calculate gaps')
    Wks_refs_log = np.array(Wks_refs_log)
    Wks_refs_log_mean = np.mean(Wks_refs_log, axis=0)
    Wks_refs_log_std = np.std(Wks_refs_log, axis=0)
    gaps = Wks_refs_log_mean - Wks_log
    print('Finish: calculate gaps')

    print('Start: select optimal k')
    s = Wks_refs_log_std * np.sqrt(1 + 1.0 / ref_num)
    idx_selected = np.where(gaps[:-1] >= gaps[1:] - s[1:])[0][0]
    k_selected = cluster_nums[idx_selected]
    print('Finish: select optimal k')

    return labels_list, Wks, Wks_refs_log_mean, gaps, s, k_selected


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


# ---common---
def intersect(arr, mask, label=None, substitution=np.nan):
    """
    reserve values in the mask and replace values out of the mask with substitution
    :param arr: numpy array
    :param mask: numpy array
    :param label:
        specify the mask value in the mask array
    :param substitution:
    :return:
    """
    assert arr.shape == mask.shape

    if label is None:
        mask_idx_mat = mask != 0
    else:
        mask_idx_mat = mask == label

    if substitution == 'min':
        substitution = np.min(arr[mask_idx_mat])
    elif substitution == 'max':
        substitution = np.max(arr[mask_idx_mat])

    new_arr = arr.copy()
    new_arr[np.logical_not(mask_idx_mat)] = substitution
    return new_arr
