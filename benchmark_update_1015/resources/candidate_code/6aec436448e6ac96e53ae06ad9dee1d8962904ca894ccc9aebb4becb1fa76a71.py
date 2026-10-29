import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set figure size
plt.figure(figsize=(9, 4.2))

# Create dummy data: Memory allocation increases with Process ID and Time Slot
process_ids = np.arange(0, 5)  # 0 to 4
time_slots = np.arange(0, 6)   # 0 to 5
X, Y = np.meshgrid(process_ids, time_slots)
Z = 2 * X + 1.5 * Y  # dummy data: increasing with both dimensions

# Left plot: Shaded
ax1 = plt.subplot(121, projection='3d')
ax1.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z).ravel(), 
          dx=0.8, dy=0.8, dz=Z.ravel(), 
          color='#1f77b4', edgecolor='#0b3d5c', linewidth=0.6, shade=True)
ax1.set_title('Memory Allocation (Shaded)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Process ID')
ax1.set_ylabel('Time Slot')
ax1.set_zlabel('Memory (MB)')
ax1.set_zlim(0, 16)
ax1.set_zticks(np.arange(0, 15, 2))
ax1.set_xticks(np.arange(0, 6))
ax1.set_yticks(np.arange(0, 7))

# Right plot: Flat
ax2 = plt.subplot(122, projection='3d')
ax2.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z).ravel(), 
          dx=0.8, dy=0.8, dz=Z.ravel(), 
          color='#c13f83', edgecolor='#65183f', linewidth=0.6, shade=True)
ax2.set_title('Memory Allocation (Flat)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Process ID')
ax2.set_ylabel('Time Slot')
ax2.set_zlabel('Memory (MB)')
ax2.set_zlim(0, 16)
ax2.set_zticks(np.arange(0, 15, 2))
ax2.set_xticks(np.arange(0, 6))
ax2.set_yticks(np.arange(0, 7))

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/2588f28c39d64c8aaca16998bc28fd59.png", bbox_inches="tight")
