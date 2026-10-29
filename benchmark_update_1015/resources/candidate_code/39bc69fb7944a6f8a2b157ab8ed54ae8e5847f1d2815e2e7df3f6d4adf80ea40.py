import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dummy Data
process_ids = np.arange(6)
time_slots = np.arange(6)
memory_values = np.arange(1, 37).reshape(6, 6)

# Create Figure
fig = plt.figure(figsize=(9, 4))

# Left Subplot (Shaded)
ax1 = fig.add_subplot(121, projection='3d')
for i in process_ids:
    for j in time_slots:
        ax1.bar3d(i, j, 0, 1, 1, memory_values[i, j], shade=True, color='tab:blue')
ax1.set_title('Memory Allocation (Shaded)')
ax1.set_xlabel('Process ID')
ax1.set_ylabel('Time Slot')
ax1.set_zlabel('Memory (MB)')
ax1.set_zlim(0, 14)

# Right Subplot (Flat)
ax2 = fig.add_subplot(122, projection='3d')
for i in process_ids:
    for j in time_slots:
        ax2.bar3d(i, j, 0, 1, 1, memory_values[i, j], shade=False, color='tab:pink')
ax2.set_title('Memory Allocation (Flat)')
ax2.set_xlabel('Process ID')
ax2.set_ylabel('Time Slot')
ax2.set_zlabel('Memory (MB)')
ax2.set_zlim(0, 14)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/2588f28c39d64c8aaca16998bc28fd59.png", bbox_inches="tight")