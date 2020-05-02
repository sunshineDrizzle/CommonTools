import numpy as np

from commontool.io import io as ct_io
from commontool.io.io import CiftiReader, save2cifti, CsvReader


class TestCiftiReader:

    def test_get_data(self):
        reader = ct_io.CiftiReader('../data/test.dlabel.nii')
        assert reader.get_data().shape == (1, 59412)
        assert reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)[0, 0] == 1
        assert reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)[0, 0] == 1

    def test_label_info(self):
        reader = ct_io.CiftiReader('../data/test.dlabel.nii')
        label_info = reader.label_info
        assert len(label_info) == 1
        label_dict = label_info[0]
        label_dict_true = {'key': [0, 1], 'label': ['None', 'test'],
                           'rgba': np.array([[0, 0, 0, 0], [1, 0, 0, 0]])}
        assert sorted(label_dict.keys()) == sorted(label_dict_true.keys())
        assert label_dict['key'] == label_dict_true['key']
        assert label_dict['label'] == label_dict_true['label']
        assert np.all(label_dict['rgba'] == label_dict_true['rgba'])


def cifti_io():
    reader1 = CiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1\HCP_S1200_GroupAvg_v1'
                          r'\HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')

    save2cifti(r'E:\tmp\HCP_S1200_997_tfMRI_FACE-AVG_level2_cohensd_hp200_s4_MSMAll.dscalar.nii',
               np.atleast_2d(reader1.get_data()[19]),
               reader1.brain_models(), reader1.map_names([19]), reader1.volume, reader1.label_tables([19]))

    reader2 = CiftiReader(r'E:\tmp\HCP_S1200_997_tfMRI_FACE-AVG_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
    data1 = reader1.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', zeroize=True)[19]
    print(data1)
    data2 = reader2.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', zeroize=True)[0]
    print(data2)
    print(np.max(data1 - data2), np.min(data1 - data2))


def merge_nifti2cifti():
    import nibabel as nib

    # get data
    data_l = nib.load(r'E:\tmp\l_vcAtlas_refine2.nii.gz').get_data().ravel()
    data_r = nib.load(r'E:\tmp\r_vcAtlas_refine2.nii.gz').get_data().ravel()

    cifti_reader = CiftiReader(r'E:\useful_things\data\HCP\HCP_S1200_GroupAvg_v1'
                               r'\HCP_S1200_997_tfMRI_FACE-AVG_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
    # get valid data
    bm_lr = cifti_reader.brain_models(['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'])
    idx2vtx_l = list(bm_lr[0].vertex_indices)
    idx2vtx_r = list(bm_lr[1].vertex_indices)
    cifti_data_l = data_l[idx2vtx_l]
    cifti_data_r = data_r[idx2vtx_r]

    # get edge_list
    cifti_data = np.c_[np.atleast_2d(cifti_data_l), np.atleast_2d(cifti_data_r)]

    save2cifti(r'E:\tmp\vcAtlas_refine.dscalar.nii',
               cifti_data, bm_lr, ['surface vcAtlas'])


def csv_io():
    reader = CsvReader('../data/statistics.csv')
    dict0 = reader.to_dict(keys=['#subjects'])
    dict1 = reader.to_dict(1, keys=['2', '1'])

    print(dict0)
    print(dict1)
