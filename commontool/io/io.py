import nibabel as nib
import numpy as np

from nibabel.cifti2 import cifti2


class CiftiReader(object):

    def __init__(self, file_path):
        self.full_data = nib.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    @property
    def brain_structures(self):
        return [_.brain_structure for _ in self.header.get_index_map(1).brain_models]

    @property
    def volume(self):
        return self.header.get_index_map(1).volume

    def brain_models(self, structures=None):
        """
        get brain model from cifti file

        Parameter:
        ---------
        structures: list of str
            Each structure corresponds to a brain model.
            If None, get all brain models.

        Return:
        ------
            brain_models: list of Cifti2BrainModel
        """
        brain_models = list(self.header.get_index_map(1).brain_models)
        if structures is not None:
            if not isinstance(structures, list):
                raise TypeError("The parameter 'structures' must be a list")
            brain_models = [brain_models[self.brain_structures.index(s)] for s in structures]
        return brain_models

    def map_names(self, rows=None):
        """
        get map names

        Parameters:
        ----------
        rows: sequence of integer
            Specify which map names should be got.
            If None, get all map names

        Return:
        ------
        map_names: list of str
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                map_names = [named_map.map_name for named_map in named_maps]
            else:
                map_names = [named_maps[i].map_name for i in rows]
        else:
            map_names = []
        return map_names

    def label_tables(self, rows=None):
        """
        get label tables

        Parameters:
        ----------
        rows: sequence of integer
            Specify which label tables should be got.
            If None, get all label tables.

        Return:
        ------
        label_tables: list of Cifti2LableTable
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                label_tables = [named_map.label_table for named_map in named_maps]
            else:
                label_tables = [named_maps[i].label_table for i in rows]
        else:
            label_tables = []
        return label_tables

    def get_data(self, structure=None, zeroize=False):
        """
        get data from cifti file

        Parameters:
        ----------
        structure: str
            One structure corresponds to one brain model.
            specify which brain structure's data should be extracted
            If None, get all structures, meanwhile ignore parameter 'zeroize'.
        zeroize: bool
            If true, get data after filling zeros for the missing vertices.

        Return:
        ------
            data: numpy array
        """

        _data = np.array(self.full_data.get_data())
        if structure is not None:
            brain_model = self.brain_models([structure])[0]
            offset = brain_model.index_offset
            count = brain_model.index_count

            if zeroize:
                n_vtx = brain_model.surface_number_of_vertices  # FIXME brain structure may not belong to surface
                data = np.zeros((_data.shape[0], n_vtx), _data.dtype)
                data[:, list(brain_model.vertex_indices)] = _data[:, offset:offset+count]
            else:
                data = _data[:, offset:offset+count]
        else:
            data = _data

        return data


def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
        """
        Save data as a cifti file
        If you just want to simply save pure data without extra information,
        you can just supply the first three parameters.

        NOTE!!!!!!
            The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
            Nibabel-2.3.0 can support for Cifti2Image indeed.

        Parameters:
        ----------
        file_path: str
            the output filename
        data: numpy array
            An array with shape (maps, values), each row is a map.
        brain_models: sequence of Cifti2BrainModel
            Each brain model is a specification of a part of the data.
            We can always get them from another cifti file header.
        map_names: sequence of str
            The sequence's indices correspond to data's row indices and label_tables.
            And its elements are maps' names.
        volume: Cifti2Volume
            The volume contains some information about subcortical voxels,
            such as volume dimensions and transformation matrix.
            If your data doesn't contain any subcortical voxel, set the parameter as None.
        label_tables: sequence of Cifti2LableTable
            Cifti2LableTable is a mapper to map label number to Cifti2Label.
            Cifti2Lable is a specification of the label, including rgba, label name and label number.
            If your data is a label data, it would be useful.
        """

        if map_names is None:
            map_names = [None] * data.shape[0]
        else:
            assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

        if label_tables is None:
            label_tables = [None] * data.shape[0]
        else:
            assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

        # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
        # and this index_map always contains some scalar information, such as named_maps.
        # We can get label_table and map_name and metadata from named_map.
        mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], 'CIFTI_INDEX_TYPE_SCALARS')
        for mn, lbt in zip(map_names, label_tables):
            named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
            mat_idx_map0.append(named_map)

        # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
        # and this index_map always contains some brain_structure information, such as brain_models and volume.
        mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
        for bm in brain_models:
            mat_idx_map1.append(bm)
        mat_idx_map1.append(volume)

        matrix = cifti2.Cifti2Matrix()
        matrix.append(mat_idx_map0)
        matrix.append(mat_idx_map1)
        header = cifti2.Cifti2Header(matrix)
        img = cifti2.Cifti2Image(data, header)
        cifti2.save(img, file_path)


if __name__ == '__main__':
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
    print(np.max(data1-data2), np.min(data1-data2))
