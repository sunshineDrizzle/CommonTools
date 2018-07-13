import nibabel as nib
import numpy as np

from nibabel.cifti2 import cifti2


class CiftiReader(object):

    def __init__(self, file_path):
        self.full_data = nib.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    def get_data(self, structure=None, zeroize=False):
        """
        get data from cifti file

        :param structure: str
            specify which brain structure's data should be extracted
            If None, get all structures, meanwhile ignore parameter 'zeroize'.
        :param zeroize: bool
            If true, get data after filling zeros for the missing vertices.
        :return: data: numpy array
        """

        _data = np.array(self.full_data.get_data())
        if structure is not None:
            try:
                brain_model = [i for i in self.header.get_index_map(1).brain_models
                               if i.brain_structure == structure][0]
            except IndexError:
                raise Exception('The structure ({}) does not exist!'.format(structure))
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


def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_table=None):
        """
        Save data as a cifti file
        If you just want to simply save pure data without extra information,
        you can just supply the first three parameters.

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
            The sequence's indices correspond to data's row indices.
            And its elements are maps' names.
        volume: Cifti2Volume
            The volume contains some information about subcortical voxels,
            such as volume dimensions and transformation matrix.
            If your data doesn't contain any subcortical voxel, set the parameter as None.
        label_table: Cifti2LableTable
            Cifti2LableTable is a mapper to map label number to Cifti2Label.
            Cifti2Lable is a specification of the label, including rgba, label name and label number.
            If your data is a label data, it would be useful.
        """

        if map_names is not None:
            assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

        # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
        # and this index_map always contains some scalar information, such as named_maps.
        # We can get label_table and map_name and metadata from named_map.
        mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], 'CIFTI_INDEX_TYPE_SCALARS')
        for mn in map_names:
            name_map = cifti2.Cifti2NamedMap(mn, label_table=label_table)
            mat_idx_map0.append(name_map)

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
        img = nib.Cifti2Image(data, header)
        img.to_filename(file_path)
