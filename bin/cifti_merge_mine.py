if __name__ == '__main__':
    import os
    import numpy as np

    from commontool.io.io import CiftiReader, save2cifti
    from commontool.algorithm.string import get_strings_by_filling

    column = 20
    target_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/rFFA_clustering/data/S1200_face-avg'
    filler_file0 = os.path.join(target_dir, 'sessid_all')
    string = '/nfs/s2/userhome/chenxiayu/workingdir/data/HCP/S1200_WM_all_cope/' \
             '{0}_tfMRI_WM_level2_hp200_s4_MSMAll.dscalar.nii'
    out_path = os.path.join(target_dir, 'S1200.1080.FACE-AVG_level2_zstat_hp200_s4_MSMAll_mine.dscalar.nii')

    with open(filler_file0) as f0:
        fillers_c0 = f0.read().splitlines()
    src_paths = get_strings_by_filling(string, [fillers_c0])

    log_file = open(os.path.join(target_dir, 'cifti_merge_log'), 'w+')
    merged_data = []
    map_names = []
    reader = None
    for fpath in src_paths:
        if not os.path.exists(fpath):
            message = 'Path-{0} does not exist!\n'.format(fpath)
            print(message, end='')
            log_file.writelines(message)
            continue

        reader = CiftiReader(fpath)
        # If don't use .copy(), the merged_data will share the same data object with data in reader.
        # As a result, the memory space occupied by the whole data in reader will be reserved.
        # But we only need one row of the whole data, so we can use the .copy() to make the element
        # avoid being a reference to the row of the whole data. Then, the whole data in reader will
        # be regarded as a garbage and collected by Garbage Collection Program.
        merged_data.append(reader.get_data()[column-1].copy())
        map_names.append(reader.map_names()[column-1])
        print('Merged:', fpath)
    if reader is None:
        message = "Can't find any valid source path\n"
        print(message, end='')
        log_file.writelines(message)
    else:
        message = 'Start save2cifti\n'
        print(message, end='')
        log_file.writelines(message)
        save2cifti(out_path, np.array(merged_data), reader.brain_models(), map_names)

    log_file.write('done')
    log_file.close()
