if __name__ == '__main__':
    import os
    from subprocess import Popen

    from commontool.algorithm.string import get_strings_by_filling

    column = '20'
    target_dir = '/home/bnucnl/cxy_workingdir/S1200_face_avg/'
    filler_file0 = os.path.join(target_dir, 'sessid')
    string = '/s3/hcp/{0}/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s4_level2_MSMAll.feat/' \
             '{0}_tfMRI_WM_level2_hp200_s4_MSMAll.dscalar.nii'
    out_path = os.path.join(target_dir, 'S1200.All.face-avg_level2_zstat_hp200_s4_MSMAll.dscalar.nii')

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
    Popen(merge_cmd,
          stderr=open(os.path.join(target_dir, 'cifti_merge_stderr'), 'w+'),
          stdout=open(os.path.join(target_dir, 'cifti_merge_stdout'), 'w+'))
    log_file.write('done')
    log_file.close()
