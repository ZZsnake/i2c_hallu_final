import matplotlib.pyplot as plt

# Define the figure and subplots
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# Define positions of red dots
points = {
    'top': (0.5, 0.8),
    'left': (0.2, 0.2),
    'right': (0.8, 0.2)
}

# Arrow configurations for each subplot
arrows = [
    [],  # A1
    [('right', 'left')],  # A2
    [('right', 'left')],  # B1
    [('top', 'left'), ('top', 'right')],  # B2
    [('left', 'top'), ('right', 'top')],  # C1
    [('left', 'top'), ('right', 'top')],  # C2
    [('left', 'right'), ('top', 'right')],  # D1
    [('left', 'top'), ('right', 'top'), ('left', 'right')],  # D2
    [('top', 'left'), ('left', 'right')],  # E1
    [('top', 'left'), ('right', 'left')],  # E2
    [('top', 'right'), ('left', 'right')],  # F1
    [('top', 'right'), ('right', 'left')],  # F2
    [('left', 'top'), ('right', 'top'), ('left', 'right')],  # G1
    [('left', 'top'), ('right', 'top'), ('right', 'left')],  # G2
    [('left', 'top'), ('right', 'top'), ('left', 'right')],  # H1
    [('left', 'top'), ('right', 'top'), ('right', 'left')],  # H2
]

# Labels for green circles
labels = [
    'A1', 'A2', 'B1', 'B2',
    'C1', 'C2', 'D1', 'D2',
    'E1', 'E2', 'F1', 'F2',
    'G1', 'G2', 'H1', 'H2'
]

# Create each subplot
for i, ax in enumerate(axes):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw red dots
    for pos in ['top', 'left', 'right']:
        x, y = points[pos]
        ax.plot(x, y, 'ro', markersize=10)
    
    # Draw arrows
    for start, end in arrows[i]:
        x_start, y_start = points[start]
        x_end, y_end = points[end]
        ax.annotate("",
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", color='blue', lw=2))
    
    # Add green circle with label
    circle = plt.Circle((0.5, 0.5), 0.1, color='green', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.text(0.5, 0.5, labels[i], ha='center', va='center', fontsize=12, color='black')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/148c74213a904740a2b625f24a275ed8.png", bbox_inches="tight")