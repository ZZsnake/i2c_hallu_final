import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath

Path = mpath.Path
codes, vertices = zip(*[
    (Path.MOVETO, [0.022, -0.12]),
    (Path.CURVE4, [-0.028, -0.045]),
    (Path.CURVE4, [-0.125, 0.085]),
    (Path.CURVE4, [-0.035, 0.082]),
    (Path.LINETO, [-0.015, 0.042]),
    (Path.CURVE4, [0.048, 0.135]),
    (Path.CURVE4, [0.085, -0.008]),
    (Path.CURVE4, [0.032, -0.024]),
    (Path.CLOSEPOLY, [0.022, -0.12])
])

geometric_shapes = [
    mpatches.Circle((0, 0), 0.12, ec="none", facecolor='#E41A1C'),
    mpatches.Rectangle((-0.03, -0.06), 0.06, 0.12, ec="none", facecolor='#377EB8'),
    mpatches.Wedge((0, 0), 0.12, 45, 300, ec="none", facecolor='#4DAF4A'),
    mpatches.RegularPolygon((0, 0), 6, radius=0.12, facecolor='#984EA3'),
    mpatches.Ellipse((0, 0), 0.24, 0.14, facecolor='#FF7F00'),
    mpatches.Arrow(-0.06, -0.06, 0.12, 0.12, width=0.12, facecolor='#FFFF33'),
    mpatches.PathPatch(mpath.Path(vertices, codes), ec="none", facecolor='#A65628'),
    mpatches.FancyBboxPatch((-0.03, -0.06), 0.06, 0.12, ec="none",
                           boxstyle=mpatches.BoxStyle("Round", pad=0.03), facecolor='#F781BF'),
    mlines.Line2D([-0.07, 0.0, 0.12], [0.06, -0.06, 0.06], lw=4, color='#999999')
]

fig, axes = plt.subplots(3, 3, figsize=(7, 7), layout="constrained")
for idx, (ax, shape) in enumerate(zip(axes.flat, geometric_shapes)):
    ax.add_artist(shape)
    ax.set_title(f"Shape {idx+1}", fontsize=10, fontfamily='serif')
    ax.set_aspect(1)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_axis_off()

plt.show()