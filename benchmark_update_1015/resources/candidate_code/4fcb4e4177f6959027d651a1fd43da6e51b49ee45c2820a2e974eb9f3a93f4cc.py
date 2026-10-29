import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Set figure dimensions
plt.figure(figsize=(9, 14))


def peaks(x, y):
    return (3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2)
            - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2)
            - (1 / 3) * np.exp(-((x + 1) ** 2) - y ** 2))

# --- Top Plot: Constant Column Parameterization ---
plt.subplot(2, 1, 1, projection='3d')
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = 10 * peaks(X / 10, Y / 10)

ax1 = plt.gca()
for j in range(0, X.shape[1], 6):
    ax1.plot(X[:, j], Y[:, j], Z[:, j], color='tab:blue', linewidth=1.2)
ax1.set_xlim([-30, 30])
ax1.set_ylim([-30, 30])
ax1.set_zlim([-60, 80])
ax1.set_title("Constant Column Parameterization", fontweight='bold')

# --- Bottom Plot: Constant Row Parameterization ---
plt.subplot(2, 1, 2, projection='3d')
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = 10 * peaks(X / 10, Y / 10)

ax2 = plt.gca()
for i in range(0, X.shape[0], 6):
    ax2.plot(X[i, :], Y[i, :], Z[i, :], color='#c02c7a', linewidth=1.2)
ax2.set_xlim([-30, 30])
ax2.set_ylim([-30, 30])
ax2.set_zlim([-60, 80])
ax2.set_title("Constant Row Parameterization", fontweight='bold')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/a9c8f9f437e24341b21ff9cc91c3e8a9.png", bbox_inches="tight")
