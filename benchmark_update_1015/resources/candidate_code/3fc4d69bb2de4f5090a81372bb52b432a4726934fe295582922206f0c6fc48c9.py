import numpy as np
import matplotlib.pyplot as plt

# Create dummy data
x = np.linspace(-6, 6, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-((X - 1)**2 + (Y - 1)**2) / 2) + np.exp(-((X + 1)**2 + (Y + 1)**2) / 2)

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Top-left subplot (diamond shape)
axs[0, 0].imshow(Z, extent=[-6, 6, -4, 4], origin='lower', cmap='viridis', aspect='auto')
axs[0, 0].set_xlim(-6, 6)
axs[0, 0].set_ylim(-4, 4)

# Top-right subplot (diamond shape)
axs[0, 1].imshow(Z, extent=[-6, 6, -4, 4], origin='lower', cmap='viridis', aspect='auto')
axs[0, 1].set_xlim(-6, 6)
axs[0, 1].set_ylim(-4, 4)

# Bottom-left subplot (rectangle shape)
axs[1, 0].imshow(Z[:, 25:75], extent=[-6, 6, -4, 4], origin='lower', cmap='viridis', aspect='auto')
axs[1, 0].set_xlim(-6, 6)
axs[1, 0].set_ylim(-4, 4)

# Bottom-right subplot (rectangle shape)
axs[1, 1].imshow(Z[:, 25:75], extent=[-6, 6, -4, 4], origin='lower', cmap='viridis', aspect='auto')
axs[1, 1].set_xlim(-6, 6)
axs[1, 1].set_ylim(-4, 4)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/26b419b4e2ed4c1aacb007011d1172c9.png", bbox_inches="tight")