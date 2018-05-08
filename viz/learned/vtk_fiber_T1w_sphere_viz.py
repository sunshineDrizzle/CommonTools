# thanks for Dipy!!!
# references:
# 1. https://lorensen.github.io/VTKExamples/site/Cxx/Interaction/PointPicker/
# 2. http://nipy.org/dipy/examples_built/viz_advanced.html#example-viz-advanced

import numpy as np
import nibabel as nib
import vtk
import vtk.util.numpy_support as ns


# ---some colormap functions copied from dipy.viz.colormap.py---
def orient2rgb(v):
    """ standard orientation 2 rgb colormap

    v : array, shape (N, 3) of vectors not necessarily normalized

    Returns
    -------

    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    --------

    >>> from dipy.viz import colormap
    >>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> c = colormap.orient2rgb(v)

    """

    if v.ndim == 1:
        orient = v
        orient = np.abs(orient / np.linalg.norm(orient))

    if v.ndim == 2:
        orientn = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(v / orientn)

    return orient


def boys2rgb(v):
    """ boys 2 rgb cool colormap

    Maps a given field of undirected lines (line field) to rgb
    colors using Boy's Surface immersion of the real projective
    plane.
    Boy's Surface is one of the three possible surfaces
    obtained by gluing a Mobius strip to the edge of a disk.
    The other two are the crosscap and Roman surface,
    Steiner surfaces that are homeomorphic to the real
    projective plane (Pinkall 1986). The Boy's surface
    is the only 3D immersion of the projective plane without
    singularities.
    Visit http://www.cs.brown.edu/~cad/rp2coloring for further details.
    Cagatay Demiralp, 9/7/2008.

    Code was initially in matlab and was rewritten in Python for dipy by
    the Dipy Team. Thank you Cagatay for putting this online.

    Parameters
    ------------
    v : array, shape (N, 3) of unit vectors (e.g., principal eigenvectors of
       tensor data) representing one of the two directions of the
       undirected lines in a line field.

    Returns
    ---------
    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    ----------

    >>> from dipy.viz import colormap
    >>> v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> c = colormap.boys2rgb(v)
    """

    if v.ndim == 1:
        x = v[0]
        y = v[1]
        z = v[2]

    if v.ndim == 2:
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]

    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2

    x3 = x * x2
    y3 = y * y2
    z3 = z * z2

    z4 = z * z2

    xy = x * y
    xz = x * z
    yz = y * z

    hh1 = .5 * (3 * z2 - 1) / 1.58
    hh2 = 3 * xz / 2.745
    hh3 = 3 * yz / 2.745
    hh4 = 1.5 * (x2 - y2) / 2.745
    hh5 = 6 * xy / 5.5
    hh6 = (1 / 1.176) * .125 * (35 * z4 - 30 * z2 + 3)
    hh7 = 2.5 * x * (7 * z3 - 3 * z) / 3.737
    hh8 = 2.5 * y * (7 * z3 - 3 * z) / 3.737
    hh9 = ((x2 - y2) * 7.5 * (7 * z2 - 1)) / 15.85
    hh10 = ((2 * xy) * (7.5 * (7 * z2 - 1))) / 15.85
    hh11 = 105 * (4 * x3 * z - 3 * xz * (1 - z2)) / 59.32
    hh12 = 105 * (-4 * y3 * z + 3 * yz * (1 - z2)) / 59.32

    s0 = -23.0
    s1 = 227.9
    s2 = 251.0
    s3 = 125.0

    ss23 = ss(2.71, s0)
    cc23 = cc(2.71, s0)
    ss45 = ss(2.12, s1)
    cc45 = cc(2.12, s1)
    ss67 = ss(.972, s2)
    cc67 = cc(.972, s2)
    ss89 = ss(.868, s3)
    cc89 = cc(.868, s3)

    X = 0.0

    X = X + hh2 * cc23
    X = X + hh3 * ss23

    X = X + hh5 * cc45
    X = X + hh4 * ss45

    X = X + hh7 * cc67
    X = X + hh8 * ss67

    X = X + hh10 * cc89
    X = X + hh9 * ss89

    Y = 0.0

    Y = Y + hh2 * -ss23
    Y = Y + hh3 * cc23

    Y = Y + hh5 * -ss45
    Y = Y + hh4 * cc45

    Y = Y + hh7 * -ss67
    Y = Y + hh8 * cc67

    Y = Y + hh10 * -ss89
    Y = Y + hh9 * cc89

    Z = 0.0

    Z = Z + hh1 * -2.8
    Z = Z + hh6 * -0.5
    Z = Z + hh11 * 0.3
    Z = Z + hh12 * -2.5

    # scale and normalize to fit
    # in the rgb space

    w_x = 4.1925
    trl_x = -2.0425
    w_y = 4.0217
    trl_y = -1.8541
    w_z = 4.0694
    trl_z = -2.1899

    if v.ndim == 2:

        N = len(x)
        C = np.zeros((N, 3))

        C[:, 0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
        C[:, 1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
        C[:, 2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

    if v.ndim == 1:

        C = np.zeros((3,))
        C[0] = 0.9 * np.abs(((X - trl_x) / w_x)) + 0.05
        C[1] = 0.9 * np.abs(((Y - trl_y) / w_y)) + 0.05
        C[2] = 0.9 * np.abs(((Z - trl_z) / w_z)) + 0.05

    return C


def line_colors(streamlines, cmap='rgb_standard'):
    """ Create colors for streamlines to be used in fvtk.line

    Parameters
    ----------
    streamlines : sequence of ndarrays
    cmap : ('rgb_standard', 'boys_standard')

    Returns
    -------
    colors : ndarray
    """

    if cmap == 'rgb_standard':
        col_list = [orient2rgb(streamline[-1] - streamline[0])
                    for streamline in streamlines]

    if cmap == 'boys_standard':
        col_list = [boys2rgb(streamline[-1] - streamline[0])
                    for streamline in streamlines]

    return np.vstack(col_list)


def colormap_lookup_table(scale_range=(0, 1), hue_range=(0.8, 0),
                          saturation_range=(1, 1), value_range=(0.8, 0.8)):
    """ Lookup table for the colormap

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the mininum
        and maximum value of your data. Default is (0, 1).
    hue_range : tuple of floats
        HSV values (min 0 and max 1). Default is (0.8, 0).
    saturation_range : tuple of floats
        HSV values (min 0 and max 1). Default is (1, 1).
    value_range : tuple of floats
        HSV value (min 0 and max 1). Default is (0.8, 0.8).

    Returns
    -------
    lookup_table : vtkLookupTable

    """
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetRange(scale_range)
    lookup_table.SetTableRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)

    lookup_table.Build()
    return lookup_table


# ---some utilities copied from dipy.viz.utils.py---
def numpy_to_vtk_colors(colors):
    """ Numpy color array to a vtk color array

    Parameters
    ----------
    colors: ndarray

    Returns
    -------
    vtk_colors : vtkDataArray

    Notes
    -----
    If colors are not already in UNSIGNED_CHAR you may need to multiply by 255.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.viz.utils import numpy_to_vtk_colors
    >>> rgb_array = np.random.rand(100, 3)
    >>> vtk_colors = numpy_to_vtk_colors(255 * rgb_array)
    """
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True,
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


# ---adapt from vtk_pickpieces.py---
class MouseInteractorStylePP1(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, renWin, actors):
        # super(MouseInteractorStylePP1, self).__init__()
        # The following three events are involved in the actors interaction.
        self.AddObserver('RightButtonPressEvent', self.OnRightButtonDown)
        self.AddObserver('RightButtonReleaseEvent', self.OnRightButtonUp)
        self.AddObserver('MouseMoveEvent', self.OnMouseMove)

        # Remember data we need for the interaction
        self.renderer = renderer
        self.chosenActor = None
        self.renWin = renWin
        self.actors = actors

    def OnRightButtonUp(self, obj, eventType):
        # When the right button is released, we stop the interaction
        self.chosenActor = None

        # Call parent interaction
        # super(MouseInteractorStylePP, self).OnRightButtonUp(self)
        vtk.vtkInteractorStyleTrackballCamera.OnRightButtonUp(self)

    def OnRightButtonDown(self, obj, eventType):
        # The rightbutton can be used to pick up the actor.

        # Get the display mouse event position
        screen_pos = self.GetInteractor().GetEventPosition()

        # Use a picker to see which actor is under the mouse
        self.GetInteractor().GetPicker().Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
        actor = self.GetInteractor().GetPicker().GetActor()

        # Is this an actor that we should interact with?
        if actor in self.actors:
            # Yes! Remember it.
            self.chosenActor = actor
            self.world_pos = self.GetInteractor().GetPicker().GetPickPosition()

        # Call parent interaction
        # super(MouseInteractorStylePP, self).OnRightButtonDown(self)
        vtk.vtkInteractorStyleTrackballCamera.OnRightButtonDown(self)

    def OnMouseMove(self, obj, eventType):
        # Translate a choosen actor
        if self.chosenActor is not None:
            # Redo the same calculation as during OnRightButtonDown
            screen_pos = self.GetInteractor().GetEventPosition()
            self.GetInteractor().GetPicker().Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
            actor_new = self.GetInteractor().GetPicker().GetActor()
            if actor_new is not self.chosenActor:
                return None
            world_pos_new = self.GetInteractor().GetPicker().GetPickPosition()

            # Calculate the xy movement
            dx = world_pos_new[0] - self.world_pos[0]
            dy = world_pos_new[1] - self.world_pos[1]
            dz = world_pos_new[2] - self.world_pos[2]

            # Remember the new reference coordinate
            self.world_pos = world_pos_new

            # Shift the choosen actor in the xy plane
            x, y, z = self.chosenActor.GetPosition()
            self.chosenActor.SetPosition(x + dx, y + dy, z + dz)

            # Request a redraw
            self.renWin.Render()
        else:
            # super(MouseInteractorStylePP, self).OnMouseMove(self)
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)


class MouseInteractorStylePP2(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, src_actor, tar_actors):
        # super(MouseInteractorStylePP2, self).__init__()
        self.AddObserver('RightButtonReleaseEvent', self.OnRightButtonUp)

        self.src_actor = src_actor
        self.tar_actors = tar_actors
        self.prop_picker = vtk.vtkPropPicker()  # vtkPointPicker seemingly can't pick the vtkImageActor
        # vtkPropPicker can't pick the vtkImageActor too

    def OnRightButtonUp(self, obj, eventType):
        # Get the display mouse event position
        screen_pos = self.GetInteractor().GetEventPosition()

        # Use a picker to see which actor is under the mouse
        self.prop_picker.Pick(screen_pos[0], screen_pos[1], 0,
                              self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer())
        actor = self.prop_picker.GetActor()
        print(self.prop_picker)
        print(actor)

        if actor in self.tar_actors:
            pos_new = self.prop_picker.GetPickPosition()
            self.src_actor.SetPosition(*pos_new)

        # Call parent interaction
        vtk.vtkInteractorStyleTrackballCamera.OnRightButtonUp(self)


# ****************************
# ***the main program entry***
# ****************************
# ===fiber visualization===
# ---get array sequence---
tck_file = nib.streamlines.load('../data/1M_20_01_20dynamic250_SD_Stream_occipital5.tck')
array_sequence = tck_file.streamlines[:10]

# ---get vtk points---
points = np.vstack(array_sequence)
vtk_points = vtk.vtkPoints()
vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))

# ---get vtk lines---
n_lines = len(array_sequence)

# Get lines_array in vtk input format
lines_array = []
# Using np.intp (instead of int64), because of a bug in numpy:
# https://github.com/nipy/dipy/pull/789
# https://github.com/numpy/numpy/issues/4384
points_per_line = np.zeros([n_lines], np.intp)
current_position = 0
for i in range(n_lines):
    current_len = len(array_sequence[i])
    points_per_line[i] = current_len

    end_position = current_position + current_len
    lines_array += [current_len]
    lines_array += range(current_position, end_position)
    current_position = end_position
lines_array = np.array(lines_array)

# Set Lines to vtk array format
vtk_lines = vtk.vtkCellArray()
vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(lines_array))
vtk_lines.SetNumberOfCells(n_lines)

# ---set automatic rgb colors---
cols_arr = line_colors(array_sequence)
# each point on the same line is assigned the same color
# the index of colors_mapper is the index of each point in vtk_points
# the element of colors_mapper is the index of each row in cols_arr
colors_mapper = np.repeat(range(n_lines), points_per_line, axis=0)
vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
vtk_colors.SetName("Colors")

# ---create the poly data---
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(vtk_points)
poly_data.SetLines(vtk_lines)
poly_data.GetPointData().SetScalars(vtk_colors)

# ---create poly mapper---
poly_mapper = vtk.vtkPolyDataMapper()
poly_mapper.SetInputData(poly_data)
poly_mapper.ScalarVisibilityOn()
poly_mapper.SetScalarModeToUsePointFieldData()
poly_mapper.SelectColorArray("Colors")
poly_mapper.Update()

# ---create actor---
slicer_opacity = 0.6
is_lod = False
if is_lod:
    # Use vtkLODActor(level of detail) rather than vtkActor.
    # Level of detail actors do not render the full geometry when the frame rate is low.
    actor = vtk.vtkLODActor()
    actor.SetNumberOfCloudPoints(10 ** 4)
    actor.GetProperty().SetPointSize(3)
else:
    actor = vtk.vtkActor()
actor.SetMapper(poly_mapper)
actor.GetProperty().SetLineWidth(1)
actor.GetProperty().SetOpacity(1)

# ===T1w image of brain in three planes' visualization===
# ---set T1w data to vtkImageData---
n_components = 1  # only support 1 color channel at present
T1_file = nib.load('../data/T1w_acpc_dc_restore_brain1.25.nii.gz')
T1_data = T1_file.get_data()
affine = T1_file.affine
vol = np.interp(T1_data, xp=[T1_data.min(), T1_data.max()], fp=[0, 255])
vol = vol.astype(np.int8)
im = vtk.vtkImageData()
I, J, K = vol.shape
im.SetDimensions(I, J, K)
voxsz = (1., 1., 1.)
im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, n_components)
vol = np.swapaxes(vol, 0, 2)
vol = np.ascontiguousarray(vol)
vol = vol.ravel()
uchar_array = ns.numpy_to_vtk(vol, deep=0)
im.GetPointData().SetScalars(uchar_array)

# ---set the transform (identity if none given)---
if affine is None:
    affine = np.eye(4)

transform = vtk.vtkTransform()
transform_matrix = vtk.vtkMatrix4x4()
transform_matrix.DeepCopy((
    affine[0][0], affine[0][1], affine[0][2], affine[0][3],
    affine[1][0], affine[1][1], affine[1][2], affine[1][3],
    affine[2][0], affine[2][1], affine[2][2], affine[2][3],
    affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
transform.SetMatrix(transform_matrix)
transform.Inverse()

# ---set the reslicing---
image_resliced = vtk.vtkImageReslice()
image_resliced.SetInputData(im)
image_resliced.SetResliceTransform(transform)
image_resliced.AutoCropOutputOn()

# Adding this will allow to support anisotropic voxels
# and also gives the opportunity to slice per voxel coordinates
RZS = affine[:3, :3]
zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
image_resliced.SetOutputSpacing(*zooms)
image_resliced.SetInterpolationModeToLinear()
image_resliced.Update()

# ---create a black/white lookup table---
lut = colormap_lookup_table((0, 255), (0, 0), (0, 0), (0, 1))
plane_colors = vtk.vtkImageMapToColors()
plane_colors.SetLookupTable(lut)
plane_colors.SetInputConnection(image_resliced.GetOutputPort())
plane_colors.Update()

# ---create vtkImageActor---
# x1, x2, y1, y2, z1, z2 = im.GetExtent()
ex1, ex2, ey1, ey2, ez1, ez2 = image_resliced.GetOutput().GetExtent()

image_actor_x = vtk.vtkImageActor()
image_actor_x.GetMapper().SetInputConnection(plane_colors.GetOutputPort())
image_actor_x.SetDisplayExtent(ex2//2, ex2//2, ey1, ey2, ez1, ez2)
image_actor_x.Update()
image_actor_x.SetInterpolate(True)
image_actor_x.GetMapper().BorderOn()
image_actor_x.GetProperty().SetOpacity(slicer_opacity)

image_actor_y = vtk.vtkImageActor()
image_actor_y.GetMapper().SetInputConnection(plane_colors.GetOutputPort())
image_actor_y.SetDisplayExtent(ex1, ex2, ey2//2, ey2//2, ez1, ez2)
image_actor_y.Update()
image_actor_y.SetInterpolate(True)
image_actor_y.GetMapper().BorderOn()
image_actor_y.GetProperty().SetOpacity(slicer_opacity)

image_actor_z = vtk.vtkImageActor()
image_actor_z.GetMapper().SetInputConnection(plane_colors.GetOutputPort())
image_actor_z.SetDisplayExtent(ex1, ex2, ey1, ey2, ez2//2, ez2//2)
image_actor_z.Update()
image_actor_z.SetInterpolate(True)
image_actor_z.GetMapper().BorderOn()
image_actor_z.GetProperty().SetOpacity(slicer_opacity)

# ===add a sphere===
# ---create sphere---
sphere_src = vtk.vtkSphereSource()
sphere_src.SetCenter(0, 0, 0)
sphere_src.SetRadius(5.0)

sphere_mapper = vtk.vtkPolyDataMapper()
sphere_mapper.SetInputConnection(sphere_src.GetOutputPort())

sphere_actor = vtk.vtkActor()
sphere_actor.SetMapper(sphere_mapper)

# ===add foregoing actors into a render window===
# ---create a renderer---
ren = vtk.vtkRenderer()
ren.AddActor(actor)
ren.AddActor(image_actor_x)
ren.AddActor(image_actor_y)
ren.AddActor(image_actor_z)
ren.AddActor(sphere_actor)
ren.ResetCamera()

# ---create a render window---
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetWindowName('VTK window made by CXY')
renWin.SetSize(800, 600)

# ---create a render window interactor for interaction---
iren = vtk.vtkRenderWindowInteractor()
# istyle = vtk.vtkInteractorStyleTrackballCamera()
# istyle = vtk.vtkInteractorStyleImage()
istyle = MouseInteractorStylePP1(ren, renWin, [sphere_actor])
# istyle = MouseInteractorStylePP2(sphere_actor, [image_actor_x, image_actor_y, image_actor_z])
istyle.SetCurrentRenderer(ren)
# Hack: below, we explicitly call the Python version of SetInteractor.
istyle.SetInteractor(iren)
point_picker = vtk.vtkPointPicker()
iren.SetPicker(point_picker)
iren.SetInteractorStyle(istyle)
iren.SetRenderWindow(renWin)
iren.Initialize()

# ---start visualizing---
renWin.Render()
iren.Start()
