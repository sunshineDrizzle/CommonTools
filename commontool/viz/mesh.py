import numpy as np

from mayavi import mlab


def show_triangular_mesh(x, y, z, triangles, scalars,
                         vmin=None, vmax=None, colormap='jet',
                         bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 350),
                         azimuth=-48, elevation=142, distance=422, focalpoint=(33, -17, 16), roll=-84,
                         cbar_orientation=None, cbar_position=None, cbar_position2=None, cbar_label_fmt=None,
                         mode='show', **kwargs):
    """

    Parameters:
    ----------
    x[ndarray]: 1-D array with shape as (n_vtx)
    y[ndarray]: 1-D array with shape as (n_vtx)
    z[ndarray]: 1-D array with shape as (n_vtx)
    triangles[seq]: a list of triplets (or an array) list the vertices in each triangle
    scalars[ndarray]: 1-D or 2-D array
        if 1-D array, its shape is (n_vtx,).
        if 2-D array, its shape is (N, n_vtx), where N is the number of maps.
            Note, 2-D is only recommended to be used with save_path!!!
            NOTE!!!, this function can't be used before find the method to set vmin and vmax outside.
    vmin[float]: the start of the colorbar
    vmax[float]: the end of the colorbar
    colormap[str]: the color map method
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
    mode[str]: show or return
        If is 'show', just show the figure.
        If is 'return', return screen shot and close the figure.
        Else, regard as the output path of the figure.
    kwargs: reference to doc of mlab.triangular_mesh

    Return:
    ------
    img[ndarray]:
    """
    if vmin is None:
        vmin = np.min(scalars)
    if vmax is None:
        vmax = np.max(scalars)
    fig = mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor, size=size)
    surf = mlab.triangular_mesh(x, y, z, triangles, scalars=scalars, figure=fig,
                                vmin=vmin, vmax=vmax, colormap=colormap, **kwargs)

    # fix the color of the smallest scalar
    lut = np.array(surf.module_manager.scalar_lut_manager.lut.table)
    lut[0, :3] = 127.5
    surf.module_manager.scalar_lut_manager.lut.table = lut

    # create color bar
    cbar = mlab.scalarbar(surf, orientation=cbar_orientation, label_fmt=cbar_label_fmt, nb_labels=5)
    cbar.scalar_bar_representation.position = cbar_position
    cbar.scalar_bar_representation.position2 = cbar_position2

    # adjust camera
    mlab.view(azimuth, elevation, distance, focalpoint, figure=fig)
    mlab.roll(roll, figure=fig)

    # save out
    if mode == 'return':
        # may have some bugs
        img = mlab.screenshot(fig)
        mlab.close(fig)
        return img
    elif mode == 'show':
        pass
    else:
        # regard as output filename
        mlab.savefig(mode, figure=fig)
        mlab.close(fig)
