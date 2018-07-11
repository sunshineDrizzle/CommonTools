

if __name__ == '__main__':
    import os
    import shutil

    filler_file0 = '/nfs/s2/userhome/chenxiayu/workingdir/test/tmp/sessid'
    proj_par = '/nfs/s2/userhome/chenxiayu/workingdir/data'
    file_in_proj = 'surface/{0}/obj.gfeat/cope1.feat/stats/rh_zstat1_1w_fracavg.mgz'
    target_par = '/nfs/s2/userhome/chenxiayu/workingdir/test/tmp'

    with open(filler_file0) as f0:
        filler0_list = f0.read().splitlines()

    log_file = open('copy_of_interest_log', 'w+')
    for filler0 in filler0_list:
        fpath = os.path.join(proj_par, file_in_proj.format(filler0))
        target_path = os.path.join(target_par, file_in_proj.format(filler0))
        target_dir = os.path.dirname(target_path)

        if not os.path.exists(fpath):
            message = 'Path-{0} does not exist!\n'.format(fpath)
            print(message, end='')
            log_file.writelines(message)
            continue
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, mode=0o777)
        shutil.copyfile(fpath, target_path)
        print('copy {0} finished'.format(filler0))
    log_file.close()
