import nibabel as nib
import numpy as np


class CiftiReader(object):

    def __init__(self, file_path):
        self.full_data = nib.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    def get_data(self, structure=None, zeroize=False):

        _data = self.full_data.get_data()
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
