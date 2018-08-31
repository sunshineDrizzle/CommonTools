if __name__ == '__main__':
    """
    NOTE1!!! wb_command -cifti-merge may load all source cifti files into memory.
    NOTE2!!! wb_command -cifti-merge may destroy partial data when merge too many files.
    I do have experienced the two phenomena above.
    """
    import os
    from subprocess import run

    from commontool.algorithm.string import get_strings_by_filling

    column = '20'
    target_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/rFFA_clustering/data/S1200_face-avg'
    filler_file0 = os.path.join(target_dir, 'sessid_all')
    string = '/nfs/s2/userhome/chenxiayu/workingdir/data/HCP/S1200_WM_all_cope/' \
             '{0}_tfMRI_WM_level2_hp200_s4_MSMAll.dscalar.nii'
    out_path = os.path.join(target_dir, 'S1200.1080.face-avg_level2_zstat_hp200_s4_MSMAll.dscalar.nii')

    with open(filler_file0) as f0:
        fillers_c0 = f0.read().splitlines()
    src_paths = get_strings_by_filling(string, [fillers_c0])

    log_file = open(os.path.join(target_dir, 'cifti_merge_log'), 'w+')
    merge_cmd = ['wb_command', '-cifti-merge', out_path]
    for fpath in src_paths:
        if not os.path.exists(fpath):
            message = 'Path-{0} does not exist!\n'.format(fpath)
            print(message, end='')
            log_file.writelines(message)
            continue
        merge_cmd.extend(['-cifti', fpath, '-column', column])

    log_file.writelines('Running: ' + ' '.join(merge_cmd) + '\n')
    run(merge_cmd,
        stderr=open(os.path.join(target_dir, 'cifti_merge_stderr'), 'w+'),
        stdout=open(os.path.join(target_dir, 'cifti_merge_stdout'), 'w+'))
    log_file.write('done')
    log_file.close()
