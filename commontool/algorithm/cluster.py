def hac_scipy(data, cluster_nums, method='ward', metric='euclidean',
              criterion='maxclust', optimal_ordering=False, out_path=None):
    """
    Perform hierarchical/agglomerative clustering on data

    :param data: see linkage
    :param cluster_nums: sequence | iterator
        Each element is the number of clusters that HAC generate.
    :param method: see linkage
    :param metric: see linkage
    :param criterion: see fcluster
    :param optimal_ordering: see linkage
    :param out_path: str
        plot hierarchical clustering as a dendrogram and save it out to the path

    :return: labels_list: list
        label results of each cluster_num
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from matplotlib import pyplot as plt

    # do hierarchical clustering on FFA_data and show the dendrogram by using scipy
    Z = linkage(data, method, metric, optimal_ordering)
    labels_list = []
    for num in cluster_nums:
        labels_list.append(fcluster(Z, num, criterion))
        print('HAC finished: {}'.format(num))

    if out_path is not None:
        plt.figure()
        dendrogram(Z)
        plt.savefig(out_path)

    return labels_list
