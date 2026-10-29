import matplotlib.pyplot as plt
import numpy as np

# Initialize figure with two 3D subplots
fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Generate synthetic data for memory allocation patterns
_x = np.arange(5)
_y = np.arange(6)
_xx, _yy = np.meshgrid(_x, _y)
x_coords, y_coords = _xx.ravel(), _yy.ravel()

# Calculate memory usage values
memory_usage = x_coords * 2 + y_coords * 1.5
base_level = np.zeros_like(memory_usage)
bar_width = bar_depth = 0.8

# Left subplot with shading
ax1.bar3d(x_coords, y_coords, base_level, bar_width, bar_depth, memory_usage, 
          shade=True, color='#2E86AB', edgecolor='#1B3B6F', linewidth=0.5)
ax1.set_title('Memory Allocation (Shaded)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Process ID', fontsize=9)
ax1.set_ylabel('Time Slot', fontsize=9)
ax1.set_zlabel('Memory (MB)', fontsize=9)

# Right subplot without shading
ax2.bar3d(x_coords, y_coords, base_level, bar_width, bar_depth, memory_usage, 
          shade=False, color='#A23B72', edgecolor='#561F37', linewidth=0.8)
ax2.set_title('Memory Allocation (Flat)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Process ID', fontsize=9)
ax2.set_ylabel('Time Slot', fontsize=9)
ax2.set_zlabel('Memory (MB)', fontsize=9)

plt.tight_layout()
plt.show()