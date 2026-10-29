import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# Figure setup
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(-5, 5)
ax.axis('off')

red = '#d32f2f'

# --- Main spine ---
# horizontal spine
ax.add_line(Line2D([1.2, 8.8], [0, 0], lw=6, color=red, solid_capstyle='round'))

# left arrowhead (triangle)
left_triangle = Polygon([[0.5, 0], [1.2, 0.7], [1.2, -0.7]],
                        closed=True, facecolor=red, edgecolor=red)
ax.add_patch(left_triangle)

# right ISSUE bubble (rounded rectangle/ellipse)
issue = FancyBboxPatch((8.6, -0.8), 1.2, 1.6,
                       boxstyle="round,pad=0.2",
                       linewidth=0, facecolor=red)
ax.add_patch(issue)
ax.text(9.2, 0, 'ISSUE', color='white', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Helper to draw category box
def category_box(x, y, text_label):
    box = Rectangle((x - 0.8, y - 0.3), 1.6, 0.6,
                    facecolor=red, edgecolor=red)
    ax.add_patch(box)
    ax.text(x, y, text_label, ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

# Helper to draw diagonal branch
def diag_branch(x0, y0, x1, y1):
    ax.add_line(Line2D([x0, x1], [y0, y1], lw=2.5, color=red))

# Helper to draw cause arrows (horizontal with arrowhead)
def cause_arrow(x_start, y, text, align='left', direction='right'):
    # arrow from text toward branch
    if direction == 'right':
        x_end = x_start + 0.9
        ax.add_patch(FancyArrow(x_start, y, 0.8, 0,
                                width=0.02, length_includes_head=True,
                                head_width=0.2, head_length=0.2,
                                color='black'))
        ax.text(x_start - 0.05, y, text, ha='right', va='center',
                fontsize=8)
    else:
        x_end = x_start - 0.9
        ax.add_patch(FancyArrow(x_start, y, -0.8, 0,
                                width=0.02, length_includes_head=True,
                                head_width=0.2, head_length=0.2,
                                color='black'))
        ax.text(x_start + 0.05, y, text, ha='left', va='center',
                fontsize=8)

# --- Upper branches ---

# MONITORING (left-top)
diag_branch(3.0, 0, 2.2, 2.3)
category_box(2.0, 2.6, 'MONITORING')
cause_arrow(2.3, 1.9, 'Error tracking', direction='right')
cause_arrow(2.3, 1.5, 'Performance metrics', direction='right')
cause_arrow(2.3, 1.1, 'User feedback', direction='right')

# DEVELOPMENT (center-top)
diag_branch(5.0, 0, 5.0, 2.3)
category_box(5.0, 2.6, 'DEVELOPMENT')
cause_arrow(5.3, 1.9, 'Integration problems', direction='right')
cause_arrow(5.3, 1.5, 'Coding errors', direction='right')
cause_arrow(5.3, 1.1, 'Testing gaps', direction='right')

# PLANNING (right-top)
diag_branch(7.0, 0, 7.8, 2.3)
category_box(8.0, 2.6, 'PLANNING')
cause_arrow(7.7, 1.9, 'Resource allocation', direction='right')
cause_arrow(7.7, 1.5, 'Budget constraints', direction='right')
cause_arrow(7.7, 1.1, 'Timeline issues', direction='right')
cause_arrow(7.7, 0.7, 'Scope changes', direction='right')

# --- Lower branches ---

# MAINTENANCE (left-bottom)
diag_branch(3.0, 0, 2.2, -2.3)
category_box(2.0, -2.6, 'MAINTENANCE')
cause_arrow(2.3, -1.9, 'Technical debt', direction='right')
cause_arrow(2.3, -2.3, 'Bug fixes', direction='right')
cause_arrow(2.3, -2.7, 'Updates', direction='right')

# DEPLOYMENT (center-bottom)
diag_branch(5.0, 0, 5.0, -2.3)
category_box(5.0, -2.6, 'DEPLOYMENT')
cause_arrow(5.3, -1.9, 'Security concerns', direction='right')
cause_arrow(5.3, -2.3, 'Server configuration', direction='right')
cause_arrow(5.3, -2.7, 'Database setup', direction='right')

# DESIGN (right-bottom)
diag_branch(7.0, 0, 7.8, -2.3)
category_box(8.0, -2.6, 'DESIGN')
cause_arrow(7.7, -1.9, 'Aesthetics', direction='right')
cause_arrow(7.7, -2.3, 'Technical limitations', direction='right')
cause_arrow(7.7, -2.7, 'User experience', direction='right')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")