import matplotlib.pyplot as plt
import numpy as np

# Create the figure and the 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 7))

# Set the main title for the figure
fig.suptitle("Geometric Transformations of Conic Sections", fontsize=16, fontweight='bold')

# Generate parameter 't' for parametric equations
t = np.linspace(0, 2 * np.pi, 200)

# --- Subplot 1: Default aspect ratio (elliptical) ---
ax = axs[0, 0]
x1 = 4.0 * np.cos(t)
y1 = 2.5 * np.sin(t)
ax.plot(x1, y1, color='C0', linestyle='-', linewidth=2)
ax.set_title('Default aspect ratio (elliptical)', fontweight='bold')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-2.8, 2.8)
ax.set_xticks(np.arange(-4, 5, 1))
ax.set_yticks(np.arange(-2, 3, 1))
# Default aspect ratio is 'auto'

# --- Subplot 2: Equal axis scaling (circular) ---
# Note: The title is reproduced as is, but the visual is an ellipse.
ax = axs[0, 1]
x2 = 4.0 * np.cos(t)
y2 = 2.5 * np.sin(t)
ax.plot(x2, y2, color='C3', linestyle='--', linewidth=2)
ax.set_title('Equal axis scaling (circular)', fontweight='bold')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-2.8, 2.8)
ax.set_xticks(np.arange(-4, 5, 1))
ax.set_yticks(np.arange(-2, 3, 1))
# Default aspect ratio is 'auto'

# --- Subplot 3: Equal scaling with modified bounds ---
ax = axs[1, 0]
x3 = 4.0 * np.cos(t)
y3 = 2.5 * np.sin(t)
ax.plot(x3, y3, color='C2', linestyle='-', linewidth=2, marker='o', markersize=2)
ax.set_title('Equal scaling with modified bounds', fontweight='bold')
ax.set_xlim(-5, 5)
ax.set_ylim(-3.2, 3.2)
ax.set_xticks(np.arange(-4, 5, 2))
ax.set_yticks(np.arange(-3, 4, 1))
ax.set_aspect('equal', adjustable='box')

# --- Subplot 4: Aspect ratio set to equal ---
ax = axs[1, 1]
x4 = 4.0 * np.cos(t)
y4 = 2.5 * np.sin(t)
ax.plot(x4, y4, color='C4', linestyle='-.', linewidth=2)
ax.set_title('Aspect ratio set to equal', fontweight='bold')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-2.9, 2.9) # Set limits to provide padding around the circle
ax.set_xticks(np.arange(-4, 5, 1))
ax.set_yticks(np.arange(-2, 3, 1))
ax.set_aspect('equal', adjustable='box')

# Adjust layout to prevent titles from overlapping
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/73f7428cad034668b9f5252624f3bbe1.png", bbox_inches="tight")
