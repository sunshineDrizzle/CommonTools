import numpy as np
from scipy import sparse


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
        specify a area where the ROI is in.
    Returns
    -------
    lists
        each index of the list represents a vertex number
        each element is a set which includes neighbors of corresponding vertex
    """
    n_vtx = np.max(faces) + 1  # get the number of vertices

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


def average_gradient(data, edge_list):
    """
    Calculate a average gradient map for scalar data on a mesh.

    Parameters:
    ----------
    data: numpy array with shape (#vertices,)
        The indices are vertices of a mesh.
        The elements are values on the vertices.
    edge_list: list
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
        avg_gradient[vtx] = np.mean([abs(data[vtx] - data[neighbor]) for neighbor in edge_list[vtx]])
    return avg_gradient
