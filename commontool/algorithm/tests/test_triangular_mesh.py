import numpy as np

from ..triangular_mesh import get_n_ring_neighbor, average_gradient, mesh2graph
from ...io.io import GiftiReader, CiftiReader, save2cifti


def test_average_gradient():

    # get faces
    faces_l = GiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1\HCP_S1200_GroupAvg_v1'
                          r'\S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii').faces
    faces_r = GiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1\HCP_S1200_GroupAvg_v1'
                          r'\S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii').faces

    # get data
    cifti_reader = CiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1'
                               r'\HCP_S1200_997_tfMRI_FACE-AVG_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
    data_l = cifti_reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)[0]
    data_r = cifti_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)[0]

    # get idx2vtx and mask
    bm_lr = cifti_reader.brain_models(['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'])
    idx2vtx_l = list(bm_lr[0].vertex_indices)
    idx2vtx_r = list(bm_lr[1].vertex_indices)
    mask_l = np.zeros(bm_lr[0].surface_number_of_vertices, np.int)
    mask_l[idx2vtx_l] = 1
    mask_r = np.zeros(bm_lr[1].surface_number_of_vertices, np.int)
    mask_r[idx2vtx_r] = 1

    # get edge_list
    edge_list_l = get_n_ring_neighbor(faces_l, mask=mask_l)
    edge_list_r = get_n_ring_neighbor(faces_r, mask=mask_r)

    # get gradient
    gradient_data_l = average_gradient(data_l, edge_list_l)[idx2vtx_l]
    gradient_data_r = average_gradient(data_r, edge_list_r)[idx2vtx_r]
    gradient_data = np.c_[np.atleast_2d(gradient_data_l), np.atleast_2d(gradient_data_r)]

    save2cifti(r'E:\tmp\HCP_S1200_997_tfMRI_SURFACE-FACE-AVG-GRADIENT_level2_cohensd_hp200_s4_MSMAll.dscalar.nii',
               gradient_data, bm_lr, ['surface face-avg gradient'])


def test_mesh2graph():
    """By the way, test the https://python-louvain.readthedocs.io/en/latest/"""
    import community
    from froi.io.io import save2nifti

    faces = GiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1\HCP_S1200_GroupAvg_v1'
                        r'\S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii').faces
    reader = CiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1'
                         r'\HCP_S1200_997_tfMRI_FACE-AVG_level2_cohensd_hp200_s'
                         r'4_MSMAll.dscalar.nii')
    data = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    mask = data.ravel() >= 0.2
    graph = mesh2graph(faces, mask=mask, vtx_signal=data.T, weight_normalization=True)
    partition = community.best_partition(graph)

    labeled_data = np.zeros(data.shape[1])
    for vtx, label in partition.items():
        labeled_data[vtx] = label + 1

    save2nifti(r'E:\tmp\face-avg_community_thr0.2.nii.gz', labeled_data)
