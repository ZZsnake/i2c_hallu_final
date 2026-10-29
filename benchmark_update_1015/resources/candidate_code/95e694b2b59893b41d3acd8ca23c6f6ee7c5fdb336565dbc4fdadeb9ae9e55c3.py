import matplotlib.pyplot as plt
import numpy as np

# Create figure with specified dimensions
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
fig.suptitle('Geometric Transformations of Conic Sections', fontsize=14, fontweight='bold')

# Define ellipse parameters (semi-major and semi-minor axes)
a = 3.5  # semi-major axis
b = 2.0  # semi-minor axis
theta = np.linspace(0, 2*np.pi, 1000)
x = a * np.cos(theta)
y = b * np.sin(theta)

# Top-left: Default aspect ratio (elliptical)
axs[0, 0].plot(x, y, color='blue', linewidth=2)
axs[0, 0].set_title('Default aspect ratio (elliptical)', fontweight='bold')
axs[0, 0].set_xlim(-4, 4)
axs[0, 0].set_ylim(-2, 2)
axs[0, 0].set_aspect('auto')  # default

# Top-right: Equal axis scaling (circular) — but not equal aspect ratio
axs[0, 1].plot(x, y, color='red', linestyle='--', linewidth=2)
axs[0, 1].set_title('Equal axis scaling (circular)', fontweight='bold')
axs[0, 1].set_xlim(-4, 4)
axs[0, 1].set_ylim(-2, 2)
axs[0, 1].set_aspect('auto')  # same as default, but title implies intent

# Bottom-left: Equal scaling with modified bounds
axs[1, 0].plot(x, y, color='green', linewidth=2)
axs[1, 0].set_title('Equal scaling with modified bounds', fontweight='bold')
axs[1, 0].set_xlim(-4, 4)
axs[1, 0].set_ylim(-3, 3)
axs[1, 0].set_aspect('auto')

# Bottom-right: Aspect ratio set to equal
axs[1, 1].plot(x, y, color='purple', linestyle=':', linewidth=2)
axs[1, 1].set_title('Aspect ratio set to equal', fontweight='bold')
axs[1, 1].set_xlim(-4, 4)
axs[1, 1].set_ylim(-2, 2)
axs[1, 1].set_aspect('equal')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/73f7428cad034668b9f5252624f3bbe1.png", bbox_inches="tight")