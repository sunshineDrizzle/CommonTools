import numpy as np

from .io import CiftiReader, save2cifti


def test_cifti_io():
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
