# thanks for Dipy!!!
# references:
# 1. https://lorensen.github.io/VTKExamples/site/Cxx/Interaction/PointPicker/
# 2. http://nipy.org/dipy/examples_built/viz_advanced.html#example-viz-advanced

import vtk
import numpy as np
import nibabel as nib

from dipy.viz import actor, window, ui, interactor
from dipy.viz.interactor import CustomInteractorStyle


class MouseInteractorStylePP(interactor.CustomInteractorStyle):
    def __init__(self, renderer, actors):
        interactor.CustomInteractorStyle.__init__(self)
        # super(MouseInteractorStylePP, self).__init__()

        # Remember data we need for the interaction
        self.peaker = vtk.vtkPointPicker()
        self.renderer = renderer
        self.chosenActor = None
        self.actors = actors

    def OnRightButtonUp(self, obj, eventType):
        # When the right button is released, we stop the interaction
        self.chosenActor = None

        # Call parent interaction
        # super(MouseInteractorStylePP, self).on_right_button_up(obj, eventType)
        interactor.CustomInteractorStyle.on_right_button_up(self, obj, eventType)

    def OnRightButtonDown(self, obj, eventType):
        # The rightbutton can be used to pick up the actor.

        # Get the display mouse event position
        screen_pos = self.GetInteractor().GetEventPosition()

        # Use a picker to see which actor is under the mouse
        # self.GetInteractor().GetPicker().Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
        # actor = self.GetInteractor().GetPicker().GetActor()
        self.picker.Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
        actor = self.picker.GetActor()

        # Is this a actor that we should interact with?
        if actor in self.actors:
            # Yes! Remember it.
            self.chosenActor = actor
            # self.world_pos = self.GetInteractor().GetPicker().GetPickPosition()
            self.world_pos = self.picker.GetPickPosition()

        # Call parent interaction
        # super(MouseInteractorStylePP, self).on_right_button_down(obj, eventType)
        interactor.CustomInteractorStyle.on_right_button_down(self, obj, eventType)

    def OnMouseMove(self, obj, eventType):
        # Translate a choosen actor
        if self.chosenActor is not None:
            # Redo the same calculation as during OnRightButtonDown
            screen_pos = self.GetInteractor().GetEventPosition()
            # self.GetInteractor().GetPicker().Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
            # actor_new = self.GetInteractor().GetPicker().GetActor()
            self.picker.Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
            actor_new = self.picker.GetActor()
            if actor_new is not self.chosenActor:
                return None
            # world_pos_new = self.GetInteractor().GetPicker().GetPickPosition()
            world_pos_new = self.picker.GetPickPosition()

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
            self.GetInteractor().GetRenderWindow().Render()
            # self.renWin.Render()
        else:
            # super(MouseInteractorStylePP, self).on_mouse_move(obj, eventType)
            interactor.CustomInteractorStyle.on_mouse_move(self, obj, eventType)

    def SetInteractor(self, interactor):
        CustomInteractorStyle.SetInteractor(self, interactor)
        # The following three events are involved in the actors interaction.
        self.RemoveObservers('RightButtonPressEvent')
        self.RemoveObservers('RightButtonReleaseEvent')
        self.RemoveObservers('MouseMoveEvent')
        self.AddObserver('RightButtonPressEvent', self.OnRightButtonDown)
        self.AddObserver('RightButtonReleaseEvent', self.OnRightButtonUp)
        self.AddObserver('MouseMoveEvent', self.OnMouseMove)


tck_file = nib.streamlines.load('../data/1M_20_01_20dynamic250_SD_Stream_occipital5.tck')
streamlines = tck_file.streamlines[:10]
T1_file = nib.load('../data/T1w_acpc_dc_restore_brain1.25.nii.gz')
data = T1_file.get_data()
affine = T1_file.affine
shape = data.shape

world_coords = True
if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

ren = window.Renderer()
stream_actor = actor.line(streamlines)

if not world_coords:
    image_actor_z = actor.slicer(data, affine=np.eye(4))
else:
    image_actor_z = actor.slicer(data, affine)
slicer_opacity = 0.6
image_actor_z.opacity(slicer_opacity)
image_actor_x = image_actor_z.copy()
x_midpoint = int(np.round(shape[0] / 2))
image_actor_x.display_extent(x_midpoint,
                             x_midpoint, 0,
                             shape[1] - 1,
                             0,
                             shape[2] - 1)

image_actor_y = image_actor_z.copy()
y_midpoint = int(np.round(shape[1] / 2))
image_actor_y.display_extent(0,
                             shape[0] - 1,
                             y_midpoint,
                             y_midpoint,
                             0,
                             shape[2] - 1)

# ---create sphere---
sphere_src = vtk.vtkSphereSource()
sphere_src.SetCenter(0, 0, 0)
sphere_src.SetRadius(5.0)

sphere_mapper = vtk.vtkPolyDataMapper()
sphere_mapper.SetInputConnection(sphere_src.GetOutputPort())

sphere_actor = vtk.vtkActor()
sphere_actor.SetMapper(sphere_mapper)

ren.add(stream_actor)
ren.add(image_actor_z)
ren.add(image_actor_x)
ren.add(image_actor_y)
ren.add(sphere_actor)

show_m = window.ShowManager(ren, size=(1200, 900), interactor_style=MouseInteractorStylePP(ren, [sphere_actor]))
show_m.initialize()

# create sliders to move the slices and change their opacity
line_slider_z = ui.LineSlider2D(min_value=0,
                                max_value=shape[2] - 1,
                                initial_value=shape[2] / 2,
                                text_template="{value:.0f}",
                                length=140)

line_slider_x = ui.LineSlider2D(min_value=0,
                                max_value=shape[0] - 1,
                                initial_value=shape[0] / 2,
                                text_template="{value:.0f}",
                                length=140)

line_slider_y = ui.LineSlider2D(min_value=0,
                                max_value=shape[1] - 1,
                                initial_value=shape[1] / 2,
                                text_template="{value:.0f}",
                                length=140)

opacity_slider = ui.LineSlider2D(min_value=0.0,
                                 max_value=1.0,
                                 initial_value=slicer_opacity,
                                 length=140)


def change_slice_z(i_ren, obj, slider):
    z = int(np.round(slider.value))
    image_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)


def change_slice_x(i_ren, obj, slider):
    x = int(np.round(slider.value))
    image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)


def change_slice_y(i_ren, obj, slider):
    y = int(np.round(slider.value))
    image_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)


def change_opacity(i_ren, obj, slider):
    slicer_opacity = slider.value
    image_actor_z.opacity(slicer_opacity)
    image_actor_x.opacity(slicer_opacity)
    image_actor_y.opacity(slicer_opacity)


line_slider_z.add_callback(line_slider_z.slider_disk,
                           "MouseMoveEvent",
                           change_slice_z)
line_slider_z.add_callback(line_slider_z.slider_line,
                           "LeftButtonPressEvent",
                           change_slice_z)

line_slider_x.add_callback(line_slider_x.slider_disk,
                           "MouseMoveEvent",
                           change_slice_x)
line_slider_x.add_callback(line_slider_x.slider_line,
                           "LeftButtonPressEvent",
                           change_slice_x)

line_slider_y.add_callback(line_slider_y.slider_disk,
                           "MouseMoveEvent",
                           change_slice_y)
line_slider_y.add_callback(line_slider_y.slider_line,
                           "LeftButtonPressEvent",
                           change_slice_y)

opacity_slider.add_callback(opacity_slider.slider_disk,
                            "MouseMoveEvent",
                            change_opacity)
opacity_slider.add_callback(opacity_slider.slider_line,
                           "LeftButtonPressEvent",
                           change_opacity)


# create text labels to identify the sliders
def build_label(text):
    label = ui.TextBlock2D()
    label.message = text
    label.font_size = 18
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = False
    label.italic = False
    label.shadow = False
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = (1, 1, 1)

    return label


line_slider_label_z = build_label(text="Z Slice")
line_slider_label_x = build_label(text="X Slice")
line_slider_label_y = build_label(text="Y Slice")
opacity_slider_label = build_label(text="Opacity")

panel = ui.Panel2D(center=(1030, 120),
                   size=(300, 200),
                   color=(1, 1, 1),
                   opacity=0.1,
                   align="right")

# create a panel to contain the sliders and labels
panel.add_element(line_slider_label_x, 'relative', (0.1, 0.75))
panel.add_element(line_slider_x, 'relative', (0.65, 0.8))
panel.add_element(line_slider_label_y, 'relative', (0.1, 0.55))
panel.add_element(line_slider_y, 'relative', (0.65, 0.6))
panel.add_element(line_slider_label_z, 'relative', (0.1, 0.35))
panel.add_element(line_slider_z, 'relative', (0.65, 0.4))
panel.add_element(opacity_slider_label, 'relative', (0.1, 0.15))
panel.add_element(opacity_slider, 'relative', (0.65, 0.2))

show_m.ren.add(panel)

# update the position of the panel using its re_align method every time the window size changes.
global size
size = ren.GetSize()


def win_callback(obj, event):
    global size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        panel.re_align(size_change)

show_m.initialize()

interactive = True  # set the variable to True to interact with the datasets in 3D.

ren.zoom(1.5)
ren.reset_clipping_range()

if interactive:
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()

else:
    window.record(ren, out_path='bundles_and_3_slices.png', size=(1200, 900),
                  reset_camera=False)
