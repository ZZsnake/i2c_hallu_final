import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions
plt.figure(figsize=(10, 8))

# Generate dummy data for top-left subplot
x1 = np.random.uniform(-4, 4, 20)
y1 = np.random.uniform(-4, 4, 20)
plt.subplot(2, 2, 1)
for i in range(len(x1)):
    for j in range(i+1, len(x1)):
        if np.random.rand() < 0.5:
            plt.arrow(x1[i], y1[i], x1[j] - x1[i], y1[j] - y1[i], head_width=0.1, head_length=0.1, fc='k', ec='k')

# Generate dummy data for top-right subplot
x2 = np.random.uniform(-2.5, 1.0, 10)
y2 = np.random.uniform(-1.0, 1.5, 10)
plt.subplot(2, 2, 2)
for i in range(len(x2)):
    for j in range(i+1, len(x2)):
        if np.random.rand() < 0.5:
            plt.arrow(x2[i], y2[i], x2[j] - x2[i], y2[j] - y2[i], head_width=0.1, head_length=0.1, fc='r', ec='r')

# Generate dummy data for bottom-left subplot
x3 = np.random.uniform(-4, 4, 20)
y3 = np.random.uniform(-4, 4, 20)
plt.subplot(2, 2, 3)
for i in range(len(x3)):
    for j in range(i+1, len(x3)):
        if np.random.rand() < 0.5:
            plt.arrow(x3[i], y3[i], x3[j] - x3[i], y3[j] - y3[i], head_width=0.1, head_length=0.1, fc='g', ec='g')
            if np.random.rand() < 0.5:
                plt.arrow(x3[i], y3[i], x3[j] - x3[i], y3[j] - y3[i], head_width=0.1, head_length=0.1, fc='y', ec='y')

# Generate dummy data for bottom-right subplot
x4 = np.random.uniform(-2.5, 1.0, 10)
y4 = np.random.uniform(-1.0, 1.5, 10)
plt.subplot(2, 2, 4)
for i in range(len(x4)):
    for j in range(i+1, len(x4)):
        if np.random.rand() < 0.5:
            plt.arrow(x4[i], y4[i], x4[j] - x4[i], y4[j] - y4[i], head_width=0.1, head_length=0.1, fc='purple', ec='purple')

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/10cf129cb63c4f41bfb3ccb2e0d3818e.png", bbox_inches="tight")