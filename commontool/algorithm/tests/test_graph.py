def test_bfs():
    from commontool.algorithm.graph import bfs
    from commontool.io.io import save2label, GiftiReader
    from commontool.algorithm.triangular_mesh import get_n_ring_neighbor

    surface = GiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                          'S1200.L.white_MSMAll.32k_fs_LR.surf.gii')
    start = 22721
    end = 24044
    edge_list = get_n_ring_neighbor(surface.faces)
    path = bfs(edge_list, start, end)
    print(len(path))
    save2label('/nfs/t3/workingshop/chenxiayu/study/FFA_clustering/data/HCP_face-avg/label/lFFA_PA4.label',
               path, surface.coords)
