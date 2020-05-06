import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from networkx import Graph


def mesh_edges(faces):
    """
    NOTE: This function is quoted from "https://github.com/BNUCNL/FreeROI"
    Returns sparse matrix with edges as an adjacency matrix

    Parameters
    ----------
    faces : array of shape [n_triangles x 3]
        The mesh faces
    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                              shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                      shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                      shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges


def get_n_ring_neighbor(faces, n=1, ordinal=False, mask=None):
    """
    get n ring neighbor from faces array

    Parameters
    ----------
    faces : numpy array
        the array of shape [n_triangles, 3]
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    mask : 1-D numpy array
        specify a area where the ROI is
        non-ROI element's value is zero
    Returns
    -------
    lists
        each index of the list represents a vertex number
        each element is a set which includes neighbors of corresponding vertex
    """
    n_vtx = np.max(faces) + 1  # get the number of vertices
    if mask is not None and np.nonzero(mask)[0].shape[0] == n_vtx:
        # In this case, the mask covers all vertices and is equal to have no mask (None).
        # So the program reset it as a None that it will save the computational cost.
        mask = None

    # find 1_ring neighbors' id for each vertex
    coo_w = mesh_edges(faces)
    csr_w = coo_w.tocsr()
    if mask is None:
        vtx_iter = range(n_vtx)
        n_ring_neighbors = [csr_w.indices[csr_w.indptr[i]:csr_w.indptr[i+1]] for i in vtx_iter]
        n_ring_neighbors = [set(i) for i in n_ring_neighbors]
    else:
        mask_id = np.nonzero(mask)[0]
        vtx_iter = mask_id
        n_ring_neighbors = [set(csr_w.indices[csr_w.indptr[i]:csr_w.indptr[i+1]])
                            if mask[i] != 0 else set() for i in range(n_vtx)]
        for vtx in vtx_iter:
            neighbor_set = n_ring_neighbors[vtx]
            neighbor_iter = list(neighbor_set)
            for i in neighbor_iter:
                if mask[i] == 0:
                    neighbor_set.discard(i)

    if n > 1:
        # find n_ring neighbors
        one_ring_neighbors = [i.copy() for i in n_ring_neighbors]
        n_th_ring_neighbors = [i.copy() for i in n_ring_neighbors]
        # if n>1, go to get more neighbors
        for i in range(n-1):
            for neighbor_set in n_th_ring_neighbors:
                neighbor_set_tmp = neighbor_set.copy()
                for v_id in neighbor_set_tmp:
                    neighbor_set.update(one_ring_neighbors[v_id])

            if i == 0:
                for v_id in vtx_iter:
                    n_th_ring_neighbors[v_id].remove(v_id)

            for v_id in vtx_iter:
                n_th_ring_neighbors[v_id] -= n_ring_neighbors[v_id]  # get the (i+2)_th ring neighbors
                n_ring_neighbors[v_id] |= n_th_ring_neighbors[v_id]  # get the (i+2) ring neighbors
    elif n == 1:
        n_th_ring_neighbors = n_ring_neighbors
    else:
        raise RuntimeError("The number of rings should be equal or greater than 1!")

    if ordinal:
        return n_th_ring_neighbors
    else:
        return n_ring_neighbors


def mesh2edge_list(faces, n=1, ordinal=False, mask=None, vtx_signal=None,
                   weight_type=('dissimilar', 'euclidean'), weight_normalization=False):
    """
    get edge_list according to mesh's geometry and vtx_signal
    The edge_list can be used to create graph or adjacent matrix

    Parameters
    ----------
    faces : a array with shape (n_triangles, 3)
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    mask : 1-D numpy array
        specify a area where the ROI is
        non-ROI element's value is zero
    vtx_signal : numpy array
        NxM array, N is the number of vertices,
        M is the number of measurements and time points.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'pearson correlation')
    weight_normalization : bool
        If it is False, do nothing.
        If it is True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.

    Returns
    -------
    row_ind : list
        row indices of edges
    col_ind : list
        column indices of edges
    edge_data : list
        edge data of the edges-zip(row_ind, col_ind)
    """
    n_ring_neighbors = get_n_ring_neighbor(faces, n, ordinal, mask)

    row_ind = [i for i, neighbors in enumerate(n_ring_neighbors) for v_id in neighbors]
    col_ind = [v_id for neighbors in n_ring_neighbors for v_id in neighbors]
    if vtx_signal is None:
        # create unweighted edges
        n_edge = len(row_ind)  # the number of edges
        edge_data = np.ones(n_edge)
    else:
        # calculate weights according to mesh's geometry and vertices' signal
        if weight_type[0] == 'dissimilar':
            if weight_type[1] == 'euclidean':
                edge_data = [pdist(vtx_signal[[i, j]], metric=weight_type[1])[0]
                             for i, j in zip(row_ind, col_ind)]
            elif weight_type[1] == 'relative_euclidean':
                edge_data = []
                for i, j in zip(row_ind, col_ind):
                    euclidean = pdist(vtx_signal[[i, j]], metric='euclidean')[0]
                    sum_ij = np.sum(abs(vtx_signal[[i, j]]))
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
            if weight_type[1] == 'pearson correlation':
                edge_data = [pearsonr(vtx_signal[i], vtx_signal[j])[0] for i, j in zip(row_ind, col_ind)]
            elif weight_type[1] == 'mean':
                edge_data = [np.mean(vtx_signal[[i, j]]) for i, j in zip(row_ind, col_ind)]
            else:
                raise RuntimeError("The weight_type-{} is not supported now!".format(weight_type))

            if weight_normalization:
                max_similar = np.max(edge_data)
                min_similar = np.min(edge_data)
                edge_data = [(simi-min_similar)/(max_similar-min_similar) for simi in edge_data]

        else:
            raise TypeError("The weight_type-{} is not supported now!".format(weight_type))

    return row_ind, col_ind, edge_data


def mesh2adjacent_matrix(faces, n=1, ordinal=False, mask=None, vtx_signal=None,
                         weight_type=('dissimilar', 'euclidean'), weight_normalization=False):
    """
    get adjacent matrix according to mesh's geometry and vtx_signal

    Parameters
    ----------
    faces : a array with shape (n_triangles, 3)
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    mask : 1-D numpy array
        specify a area where the ROI is
        non-ROI element's value is zero
    vtx_signal : numpy array
        NxM array, N is the number of vertices,
        M is the number of measurements and time points.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'pearson correlation')
    weight_normalization : bool
        If it is False, do nothing.
        If it is True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.

    Returns
    -------
    adjacent_matrix : coo matrix
    """
    n_vtx = np.max(faces) + 1
    row_ind, col_ind, edge_data = mesh2edge_list(faces, n, ordinal, mask, vtx_signal,
                                                 weight_type, weight_normalization)
    adjacent_matrix = sparse.coo_matrix((edge_data, (row_ind, col_ind)), (n_vtx, n_vtx))

    return adjacent_matrix


def mesh2graph(faces, n=1, ordinal=False, mask=None, vtx_signal=None,
               weight_type=('dissimilar', 'euclidean'), weight_normalization=True):
    """
    create graph according to mesh's geometry and vtx_signal

    Parameters
    ----------
    faces : a array with shape (n_triangles, 3)
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    mask : 1-D numpy array
        specify a area where the ROI is
        non-ROI element's value is zero
    vtx_signal : numpy array
        NxM array, N is the number of vertices,
        M is the number of measurements and time points.
    weight_type : (str1, str2)
        The rule used for calculating weights
        such as ('dissimilar', 'euclidean') and ('similar', 'pearson correlation')
    weight_normalization : bool
        If it is False, do nothing.
        If it is True, normalize weights to [0, 1].
            After doing this, greater the weight is, two vertices of the edge are more related.

    Returns
    -------
    graph : nx.Graph
    """
    row_ind, col_ind, edge_data = mesh2edge_list(faces, n, ordinal, mask, vtx_signal,
                                                 weight_type, weight_normalization)
    graph = Graph()
    # Actually, add_weighted_edges_from is only used to add edges. If we intend to create graph by the method only,
    # all of the graph's nodes must have at least one edge. However, maybe some special graphs contain nodes
    # which have no edge connected. So we need add extra nodes.
    if mask is None:
        n_vtx = np.max(faces) + 1
        graph.add_nodes_from(range(n_vtx))
    else:
        vertices = np.nonzero(mask)[0]
        graph.add_nodes_from(vertices)

    # add_weighted_edges_from is faster than from_scipy_sparse_matrix and from_numpy_matrix
    # add_weighted_edges_from is also faster than default constructor
    # To get more related information, please refer to
    # http://stackoverflow.com/questions/24681677/transform-csr-matrix-into-networkx-graph
    graph.add_weighted_edges_from(zip(row_ind, col_ind, edge_data))

    return graph


def average_gradient(data, neighbors):
    """
    Calculate a average gradient map for scalar data on a mesh.

    Parameters:
    ----------
    data: numpy array with shape (#vertices,)
        The indices are vertices of a mesh.
        The elements are values on the vertices.
    neighbors: list
        The indices are vertices of a mesh.
        One index's corresponding element is a collection of vertices which connect to the index.

    Return:
    ------
    avg_gradient: numpy array with shape (#vertices,)
        The indices are vertices of a mesh.
        The elements are average gradient on the vertices.
    """
    avg_gradient = np.zeros_like(data)
    for vtx in range(data.shape[0]):
        avg_gradient[vtx] = np.mean([abs(data[vtx] - data[neighbor]) for neighbor in neighbors[vtx]])
    return avg_gradient


def label_edge_detection(data, faces, edge_type="inner", neighbors=None):
    """
    edge detection for labels

    Parameters
    ----------
    data : 1-D numpy array
        Each array index is corresponding to vertex id in the faces.
        Each element is a label id.
    faces : numpy array
        the array of shape [n_triangles, 3]
    edge_type : str
        "inner" means inner edges of labels.
        "outer" means outer edges of labels.
        "both" means both of them in one array
        "split" means returning inner and outer edges in two arrays respectively
    neighbors : list
        If this parameter is not None, a parameters ('faces') will be ignored.
        It is used to save time when someone repeatedly uses the function with
            a same neighbors which can be got by get_n_ring_neighbor.
        The indices are vertices' id of a mesh.
        One index's corresponding element is a collection of vertices which connect with the index.

    Return
    ------
    inner_data : 1-D numpy array
        the inner edges of the labels
    outer_data : 1-D numpy array
        the outer edges of the labels
        It's worth noting that outer_data's element values may
            be not strictly corresponding to labels' id when
            there are some labels which are too close.
    """
    # data preparation
    vertices = np.nonzero(data)[0]
    inner_data = np.zeros_like(data)
    outer_data = np.zeros_like(data)
    if neighbors is None:
        neighbors = get_n_ring_neighbor(faces)

    # look for edges
    for v_id in vertices:
        neighbors_values = [data[_] for _ in neighbors[v_id]]
        if min(neighbors_values) != max(neighbors_values):
            if edge_type in ("inner", "both", "split"):
                inner_data[v_id] = data[v_id]
            if edge_type in ("outer", "both", "split"):
                outer_vtx = [vtx for vtx in neighbors[v_id] if data[v_id] != data[vtx]]
                outer_data[outer_vtx] = data[v_id]

    # return results
    if edge_type == "inner":
        return inner_data
    elif edge_type == "outer":
        return outer_data
    elif edge_type == "both":
        return inner_data + outer_data
    elif edge_type == "split":
        return inner_data, outer_data
    else:
        raise ValueError("The argument 'edge_type' must be one of the (inner, outer, both, split)")
