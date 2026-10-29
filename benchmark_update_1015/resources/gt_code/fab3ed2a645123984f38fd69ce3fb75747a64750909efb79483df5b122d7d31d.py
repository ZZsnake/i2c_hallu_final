import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backend_bases import MouseButton
from matplotlib.patches import PathPatch
from matplotlib.path import Path

fig, ax = plt.subplots()

pathdata = [
    (Path.MOVETO, (1.25, -2.15)),
    (Path.CURVE4, (0.45, -1.35)),
    (Path.CURVE4, (-1.85, 1.75)),
    (Path.CURVE4, (0.25, 1.85)),
    (Path.LINETO, (0.65, 1.05)),
    (Path.CURVE4, (2.05, 2.95)),
    (Path.CURVE4, (2.85, 0.15)),
    (Path.CURVE4, (1.75, -0.35)),
    (Path.CLOSEPOLY, (1.25, -2.15)),
]

codes, vertices = zip(*pathdata)
curve_path = Path(vertices, codes)
curve_patch = PathPatch(
    curve_path, facecolor='blue', edgecolor='red', alpha=0.6, linewidth=2.5)
ax.add_patch(curve_patch)


class PathEditor:
    """
    Interactive path editor for computer graphics applications.
    Press 'v' to toggle control point visibility. Draggable when visible.
    """

    show_vertices = True
    selection_distance = 6

    def __init__(self, path_patch):
        self.ax = path_patch.axes
        self.canvas = self.ax.figure.canvas
        self.path_patch = path_patch
        self.path_patch.set_animated(True)

        x_coords, y_coords = zip(*self.path_patch.get_path().vertices)

        self.vertex_markers, = ax.plot(
            x_coords, y_coords, marker='s', markerfacecolor='orange', 
            markeredgecolor='black', markersize=8, animated=True)

        self.active_vertex = None

        self.canvas.mpl_connect('draw_event', self.handle_draw)
        self.canvas.mpl_connect('button_press_event', self.handle_press)
        self.canvas.mpl_connect('key_press_event', self.handle_keypress)
        self.canvas.mpl_connect('button_release_event', self.handle_release)
        self.canvas.mpl_connect('motion_notify_event', self.handle_motion)

    def find_closest_vertex(self, event):
        """Find the vertex index closest to the mouse cursor position."""
        vertices = self.path_patch.get_path().vertices
        transformed = self.path_patch.get_transform().transform(vertices)
        x_trans, y_trans = transformed[:, 0], transformed[:, 1]
        distances = np.sqrt((x_trans - event.x)**2 + (y_trans - event.y)**2)
        closest_idx = distances.argmin()
        return closest_idx if distances[closest_idx] < self.selection_distance else None

    def handle_draw(self, event):
        """Handle canvas draw events."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.path_patch)
        self.ax.draw_artist(self.vertex_markers)

    def handle_press(self, event):
        """Handle mouse press events."""
        if (event.inaxes is None or 
            event.button != MouseButton.LEFT or 
            not self.show_vertices):
            return
        self.active_vertex = self.find_closest_vertex(event)

    def handle_release(self, event):
        """Handle mouse release events."""
        if event.button != MouseButton.LEFT or not self.show_vertices:
            return
        self.active_vertex = None

    def handle_keypress(self, event):
        """Handle keyboard events."""
        if not event.inaxes:
            return
        if event.key == 'v':
            self.show_vertices = not self.show_vertices
            self.vertex_markers.set_visible(self.show_vertices)
            if not self.show_vertices:
                self.active_vertex = None
        self.canvas.draw()

    def handle_motion(self, event):
        """Handle mouse movement events."""
        if (self.active_vertex is None or
            event.inaxes is None or
            event.button != MouseButton.LEFT or
            not self.show_vertices):
            return

        vertices = self.path_patch.get_path().vertices
        vertices[self.active_vertex] = event.xdata, event.ydata
        self.vertex_markers.set_data(zip(*vertices))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.path_patch)
        self.ax.draw_artist(self.vertex_markers)
        self.canvas.blit(self.ax.bbox)


editor = PathEditor(curve_patch)
ax.set_title('Interactive Bézier Curve Editor - Drag Control Points')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

plt.show()