import numpy as np

from mayavi import mlab


def show_triangular_mesh(x, y, z, triangles, scalars,
                         vmin=None, vmax=None, colormap='jet',
                         bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 350),
                         azimuth=-48, elevation=142, distance=422, focalpoint=(33, -17, 16), roll=-84,
                         cbar_orientation=None, cbar_position=None, cbar_position2=None, cbar_label_fmt=None,
                         cbar_num=0, mode='show', **kwargs):
    """

    Parameters:
    ----------
    x[ndarray]: 1-D array with shape as (n_vtx)
    y[ndarray]: 1-D array with shape as (n_vtx)
    z[ndarray]: 1-D array with shape as (n_vtx)
    triangles[seq]: a list of triplets (or an array) list the vertices in each triangle
    scalars[ndarray]: 1-D or 2-D array
        if 1-D array, its shape is (n_vtx,).
        if 2-D array, its shape is (N, n_vtx), where N is the number of overlays.
            Overlays with larger row numbers will cover on the overlays with smaller row numbers.
            Vertices with the smallest scalar will be set transparent except for the bottom overlay.
    vmin[float|list]: the minimal scalar to display
        If is list, one-by-one corresponding to the rows of scalars.
        Else, applied to all overlays.
    vmax[float|list]: the maximal scalar to display
        If is list, one-by-one corresponding to the rows of scalars.
        Else, applied to all overlays.
    colormap[str|list]: the color map method
        If is list, one-by-one corresponding to the rows of scalars.
        Else, applied to all overlays.
    bgcolor[seq]: the rgb color of background.
    fgcolor[seq]: the rgb color of foreground.
    size[seq]: size of figure (width x height)
    azimuth[float]:
    elevation[float]:
    distance[float]:
    focalpoint[seq]:
    roll[float]:
    cbar_orientation[str]: the orientation of colorbar
    cbar_position[seq]: position of bottom-left corner
    cbar_position2[seq]: distance from the bottom-left corner
    cbar_label_fmt[str]: string format of labels of the colorbar
    cbar_num[int]: show colorbar of the cbar_num overlay
        If is 0, no colorbar will be displayed.
        If is 1, show the first overlay's colorbar, i.e. the bottom overlay.
    mode[str]: show or return
        If is 'show', just show the figure.
        If is 'return', return screen shot and close the figure.
        Else, regard as the output path of the figure.
    kwargs: reference to doc of mlab.triangular_mesh

    Return:
    ------
    img[ndarray]:
    """
    # transform x, y, z
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
    assert x.shape == y.shape and y.shape == z.shape

    # transform scalars
    if scalars.ndim == 1:
        scalars = scalars[None, :]
    elif scalars.ndim != 2:
        raise ValueError('Unsupported dimension number of scalars:', scalars.ndim)
    assert scalars.shape[1] == x.shape[0]
    n_overlay = scalars.shape[0]

    # transform vmin
    if not isinstance(vmin, list):
        vmin = [vmin] * n_overlay
    else:
        assert len(vmin) == n_overlay

    # transform vmax
    if not isinstance(vmax, list):
        vmax = [vmax] * n_overlay
    else:
        assert len(vmax) == n_overlay

    # transform colormap
    if not isinstance(colormap, list):
        colormap = [colormap] * n_overlay
    else:
        assert len(colormap) == n_overlay

    bottom_flag = True
    fig = mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor, size=size)
    for idx, data in enumerate(zip(scalars, vmin, vmax, colormap), 1):
        s, vi, va, cm = data
        if vi is None:
            vi = np.min(s)
        if va is None:
            va = np.max(s)
        surf = mlab.triangular_mesh(x, y, z, triangles, scalars=s, figure=fig,
                                    vmin=vi, vmax=va, colormap=cm, **kwargs)

        # fix the color of the smallest scalar
        lut = np.array(surf.module_manager.scalar_lut_manager.lut.table)
        if idx == 1:
            lut[0, :3] = 127.5
        else:
            lut[0, 3] = 0
        surf.module_manager.scalar_lut_manager.lut.table = lut

        if idx == cbar_num:
            # create color bar
            cbar = mlab.scalarbar(surf, orientation=cbar_orientation, label_fmt=cbar_label_fmt, nb_labels=5)
            cbar.scalar_bar_representation.position = cbar_position
            cbar.scalar_bar_representation.position2 = cbar_position2

    # adjust camera
    mlab.view(azimuth, elevation, distance, focalpoint, figure=fig)
    mlab.roll(roll, figure=fig)

    if mode == 'return':
        # may have some bugs
        img = mlab.screenshot(fig)
        mlab.close(fig)
        return img
    elif mode == 'show':
        mlab.show()
    else:
        # regard as output filename
        mlab.savefig(mode, figure=fig)
        mlab.close(fig)
