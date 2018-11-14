import numpy as np
import networkx as nx

from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def array2edge_list(array, weight_type=('dissimilar', 'euclidean'), weight_normalization=True, edges=None):
    """
    get edge_list according to the relationship between each two rows
    The edge_list can be used to create graph or adjacent matrix.

    Parameters
    ----------
    array : numpy array
        NxM array, N is the number of vertices and each row is the signal of that vertex.
        M is the length of each vertex's signal.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'correlation')
    weight_normalization : bool
        If False, do nothing.
        If True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.
    edges : str | collection
        If None, the edge_list contains the complete connections of the vertices.
        elif str, currently support 'upper_right_triangle'. 'upper_right_triangle': limit edges in the upper right
        triangle (don't include the main diagonal).
        else, regarded as a collection, each element is an edge of the two vertices. And the edge_list's edges will
        be limited in it.

    Returns
    -------
    row_ind : list
        row indices of edges
    col_ind : list
        column indices of edges
    edge_data : list
        edge data of the edges-zip(row_ind, col_ind)
    """
    # get edges' row indices and column indices
    row_ind = []
    col_ind = []
    if edges is None:
        n_vtx = array.shape[0]
        for r in range(n_vtx):
            for c in range(n_vtx):
                row_ind.append(r)
                col_ind.append(c)
    elif isinstance(edges, str):
        if edges == 'upper_right_triangle':
            n_vtx = array.shape[0]
            for i in range(n_vtx):
                for j in range(i+1, n_vtx):
                    row_ind.append(i)
                    col_ind.append(j)
        else:
            raise ValueError("The edges type-{} is not supported at present!".format(edges))
    else:
        for edge in edges:
            row_ind.append(edge[0])
            col_ind.append(edge[1])

    # calculate edge weights
    if weight_type[0] == 'dissimilar':
        if weight_type[1] == 'euclidean':
            edge_data = [pdist(array[[i, j]], metric=weight_type[1])[0]
                         for i, j in zip(row_ind, col_ind)]
        elif weight_type[1] == 'relative_euclidean':
            edge_data = []
            for i, j in zip(row_ind, col_ind):
                euclidean = pdist(array[[i, j]], metric='euclidean')[0]
                sum_ij = np.sum(abs(array[[i, j]]))
                if sum_ij:
                    edge_data.append(float(euclidean) / sum_ij)
                else:
                    edge_data.append(0)
        else:
            raise RuntimeError("The weight_type-{} is not supported now!".format(weight_type))

        if weight_normalization:
            max_dissimilar = np.max(edge_data)
            min_dissimilar = np.min(edge_data)
            edge_data = [(max_dissimilar-dist)/(max_dissimilar-min_dissimilar) for dist in edge_data]

    elif weight_type[0] == 'similar':
        if weight_type[1] == 'correlation':
            edge_data = [pearsonr(array[i], array[j])[0] for i, j in zip(row_ind, col_ind)]
        elif weight_type[1] == 'mean':
            edge_data = [np.mean(array[[i, j]]) for i, j in zip(row_ind, col_ind)]
        else:
            raise RuntimeError("The weight_type-{} is not supported now!".format(weight_type))

        if weight_normalization:
            max_similar = np.max(edge_data)
            min_similar = np.min(edge_data)
            edge_data = [(simi-min_similar)/(max_similar-min_similar) for simi in edge_data]

    else:
        raise TypeError("The weight_type-{} is not supported now!".format(weight_type))

    return row_ind, col_ind, edge_data


def array2adjacent_matrix(array, weight_type=('dissimilar', 'euclidean'), weight_normalization=True, edges=None):
    """
    create adjacent matrix according to the relationship between each two rows

    Parameters
    ----------
    array : numpy array
        NxM array, N is the number of vertices and each row is the signal of that vertex.
        M is the length of each vertex's signal.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'correlation')
    weight_normalization : bool
        If False, do nothing.
        If True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.
    edges : str | collection
        If None, the edge_list contains the complete connections of the vertices.
        elif str, currently support 'upper_right_triangle'. 'upper_right_triangle': limit edges in the upper right
        triangle (don't include the main diagonal).
        else, regarded as a collection, each element is an edge of the two vertices. And the edge_list's edges will
        be limited in it.

    Returns
    -------
    adjacent_matrix : coo matrix
    """
    n_vtx = array.shape[0]
    row_ind, col_ind, edge_data = array2edge_list(array, weight_type, weight_normalization, edges)
    adjacent_matrix = sparse.coo_matrix((edge_data, (row_ind, col_ind)), (n_vtx, n_vtx))

    return adjacent_matrix


def array2graph(array, weight_type=('dissimilar', 'euclidean'), weight_normalization=True, edges=None):
    """
    create graph according to the relationship between each two rows

    Parameters
    ----------
    array : numpy array
        NxM array, N is the number of vertices and each row is the signal of that vertex.
        M is the length of each vertex's signal.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'correlation')
    weight_normalization : bool
        If False, do nothing.
        If True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.
    edges : str | collection
        If None, the edge_list contains the complete connections of the vertices.
        elif str, currently support 'upper_right_triangle'. 'upper_right_triangle': limit edges in the upper right
        triangle (don't include the main diagonal).
        else, regarded as a collection, each element is an edge of the two vertices. And the edge_list's edges will
        be limited in it.

    Returns
    -------
    graph : nx.Graph
    """
    row_ind, col_ind, edge_data = array2edge_list(array, weight_type, weight_normalization, edges)

    graph = nx.Graph()
    # Actually, add_weighted_edges_from is only used to add edges. If we intend to create graph by the method only,
    # all of the graph's nodes must have at least one edge. However, maybe some special graphs contain nodes
    # which have no edge connected. So we need add extra nodes.
    graph.add_nodes_from(range(array.shape[0]))

    # add_weighted_edges_from is faster than from_scipy_sparse_matrix and from_numpy_matrix
    # add_weighted_edges_from is also faster than default constructor
    # To get more related information, please refer to
    # http://stackoverflow.com/questions/24681677/transform-csr-matrix-into-networkx-graph
    graph.add_weighted_edges_from(zip(row_ind, col_ind, edge_data))

    return graph
