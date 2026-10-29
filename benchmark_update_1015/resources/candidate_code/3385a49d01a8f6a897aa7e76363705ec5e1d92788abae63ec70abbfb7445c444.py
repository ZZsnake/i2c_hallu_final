import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set figure size
plt.figure(figsize=(9, 4))

# Create dummy data: Memory allocation increases with Process ID and Time Slot
process_ids = np.arange(1, 6)  # 1 to 5
time_slots = np.arange(0, 6)   # 0 to 5
X, Y = np.meshgrid(process_ids, time_slots)
Z = (X + Y) * 1.5  # dummy data: increasing with both dimensions

# Left plot: Shaded
ax1 = plt.subplot(121, projection='3d')
ax1.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z).ravel(), 
          dx=0.8, dy=0.8, dz=Z.ravel(), 
          color=plt.cm.Blues(Z.ravel() / Z.max()), shade=True)
ax1.set_title('Memory Allocation (Shaded)')
ax1.set_xlabel('Process ID')
ax1.set_ylabel('Time Slot')
ax1.set_zlabel('Memory (MB)')
ax1.set_zlim(0, 14)
ax1.set_xticks(process_ids)
ax1.set_yticks(time_slots)

# Right plot: Flat
ax2 = plt.subplot(122, projection='3d')
ax2.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z).ravel(), 
          dx=0.8, dy=0.8, dz=Z.ravel(), 
          color='purple', shade=False)
ax2.set_title('Memory Allocation (Flat)')
ax2.set_xlabel('Process ID')
ax2.set_ylabel('Time Slot')
ax2.set_zlabel('Memory (MB)')
ax2.set_zlim(0, 14)
ax2.set_xticks(process_ids)
ax2.set_yticks(time_slots)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/2588f28c39d64c8aaca16998bc28fd59.png", bbox_inches="tight")