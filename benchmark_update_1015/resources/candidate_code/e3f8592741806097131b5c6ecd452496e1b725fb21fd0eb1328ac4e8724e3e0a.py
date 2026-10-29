import matplotlib.pyplot as plt

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Use a single blue color for all elements
blue = 'blue'
lw = 1.5

# Define 2D coordinates (x, y) for the key vertices to mimic the perspective
# Front-bottom rectangle
A = (0, 0)      # front-left bottom
B = (6, 0)      # front-right bottom
E = (7.5, 1.5)  # back-right bottom
C = (1.5, 1.5)  # back-center/bottom-ish (as shown for label C)
# To better match perspective, we'll also define a back-left bottom
F = (-1.0, 1.5)  # back-left bottom (not labeled in original but needed for lines)

# Top rectangle (shift upward)
z_height = 6
D = (0, z_height)      # front-left top (labeled D)
B_top = (6, z_height)  # front-right top
E_top = (7.5, z_height + 1.5)  # back-right top
F_top = (-1.0, z_height + 1.5) # back-left top

# Internal vertical line mid-right (projection from C's x to top back position)
# In the image, vertical mid-right line is closer to B/E side, but aligned with C horizontally.
mid_bottom = (C[0] + (E[0]-C[0])*0.75, C[1] + (E[1]-C[1])*0.0)  # base of vertical internal
mid_top    = (mid_bottom[0], mid_bottom[1] + z_height)          # top of vertical internal

# Internal top horizontal line from D to mid_top
# We align this with D's vertical level but through point above C-ish
internal_top_left = (D[0] + (C[0]-A[0]), D[1] - (A[1]-C[1]))  # point roughly over C at top
internal_top_right = mid_top

# Draw outer edges of the cuboid (front rectangle)
ax.plot([A[0], B[0]], [A[1], B[1]], color=blue, lw=lw)           # bottom front AB
ax.plot([A[0], D[0]], [A[1], D[1]], color=blue, lw=lw)           # left front vertical AD
ax.plot([B[0], B_top[0]], [B[1], B_top[1]], color=blue, lw=lw)   # right front vertical
ax.plot([D[0], B_top[0]], [D[1], B_top[1]], color=blue, lw=lw)   # top front DB_top

# Back rectangle
ax.plot([F[0], E[0]], [F[1], E[1]], color=blue, lw=lw)           # back bottom
ax.plot([F_top[0], E_top[0]], [F_top[1], E_top[1]], color=blue, lw=lw)  # back top
ax.plot([F[0], F_top[0]], [F[1], F_top[1]], color=blue, lw=lw)   # back left vertical
ax.plot([E[0], E_top[0]], [E[1], E_top[1]], color=blue, lw=lw)   # back right vertical

# Connecting edges between front and back
ax.plot([A[0], F[0]], [A[1], F[1]], color=blue, lw=lw)           # bottom left depth
ax.plot([B[0], E[0]], [B[1], E[1]], color=blue, lw=lw)           # bottom right depth
ax.plot([D[0], F_top[0]], [D[1], F_top[1]], color=blue, lw=lw)   # top left depth
ax.plot([B_top[0], E_top[0]], [B_top[1], E_top[1]], color=blue, lw=lw)  # top right depth

# Internal lines
# Bottom internal diagonal from A to C
ax.plot([A[0], C[0]], [A[1], C[1]], color=blue, lw=lw)
# Internal top front-to-back (approx from over C at front-left top-ish to mid_top)
ax.plot([internal_top_left[0], internal_top_right[0]],
        [internal_top_left[1], internal_top_right[1]],
        color=blue, lw=lw)
# Internal vertical mid-right
ax.plot([mid_bottom[0], mid_top[0]],
        [mid_bottom[1], mid_top[1]],
        color=blue, lw=lw)

# Optionally draw vertical line from internal_top_left down to C to emphasize mid-plane
ax.plot([internal_top_left[0], C[0]],
        [internal_top_left[1], C[1]],
        color=blue, lw=lw)

# Vertex dots (small, to mimic endpoints)
for pt in [A, B, C, D, E, F, B_top, D, E_top, F_top]:
    ax.plot(pt[0], pt[1], 'o', color=blue, markersize=3)

# Add labels for vertices and edges
ax.text(A[0]-0.2, A[1]-0.3, 'A', color=blue, fontsize=12)
ax.text(B[0]+0.1, B[1]-0.3, 'B', color=blue, fontsize=12)
ax.text(C[0], C[1]-0.3, 'C', color=blue, fontsize=12)
ax.text(D[0]-0.3, D[1]+0.2, 'D', color=blue, fontsize=12)

# Edge labels x, y, z
# x along AB
ax.text((A[0]+B[0])/2, A[1]-0.4, 'x', color=blue, fontsize=12)
# y along AC
ax.text((A[0]+C[0])/2+0.05, (A[1]+C[1])/2+0.1, 'y', color=blue, fontsize=12)
# z along AD
ax.text(A[0]-0.4, (A[1]+D[1])/2, 'z', color=blue, fontsize=12)

# Label S somewhere near the right-top interior
S_pos = ( (B_top[0]+E_top[0])/2 - 0.4, (B_top[1]+E_top[1])/2 - 1.0 )
ax.text(S_pos[0], S_pos[1], 'S', color=blue, fontsize=12)

# Style: remove axes, ticks, and frame
ax.set_xlim(-2, 9)
ax.set_ylim(-1, 9)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/AsymptoteGraphicPipeline_Coordinate_system_0-97.png", bbox_inches="tight")