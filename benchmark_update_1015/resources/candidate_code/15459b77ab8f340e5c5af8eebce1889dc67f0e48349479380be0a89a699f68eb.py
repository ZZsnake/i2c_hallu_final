import matplotlib.pyplot as plt
import numpy as np

# Dummy data
x1 = np.array([0.70, 0.75, 0.80, 0.85, 0.90])
y1 = np.array([76, 78, 80, 78, 76])
y1_err = np.array([2, 2, 2, 2, 2])

x2 = np.array([0.1, 0.2, 0.3, 0.4])
y2 = np.array([74, 78, 76, 72])
y2_err = np.array([2, 2, 2, 2])

x3 = np.array([400, 600, 800, 1000, 1200])
y3 = np.array([80, 82.5, 77.5, 72.5, 70])
y3_err = np.array([2.5, 2.5, 2.5, 2.5, 2.5])

x4 = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y4 = np.array([75, 78, 70, 65, 60, 55])
y4_err = np.array([2, 2, 2, 2, 2, 2])

# Create figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

# Subplot 1: Positive bound
axs[0].plot(x1, y1, 'b-', label='Positive bound')
axs[0].fill_between(x1, y1 - y1_err, y1 + y1_err, color='b', alpha=0.2)
axs[0].set_xticks(x1)
axs[0].set_yticks(np.arange(70, 86, 2))
axs[0].grid(True)
axs[0].set_title('(a) Positive bound.')

# Subplot 2: Negative bound
axs[1].plot(x2, y2, 'b-', label='Negative bound')
axs[1].fill_between(x2, y2 - y2_err, y2 + y2_err, color='b', alpha=0.2)
axs[1].set_xticks(x2)
axs[1].grid(True)
axs[1].set_title('(b) Negative bound.')

# Subplot 3: Fuzzy coefficient
axs[2].plot(x3, y3, 'b-', label='Fuzzy coefficient')
axs[2].fill_between(x3, y3 - y3_err, y3 + y3_err, color='b', alpha=0.2)
axs[2].set_xticks(x3)
axs[2].grid(True)
axs[2].set_title('(c) Fuzzy coefficient.')

# Subplot 4: Contrastive loss weight
axs[3].plot(x4, y4, 'b-', label='Contrastive loss weight')
axs[3].fill_between(x4, y4 - y4_err, y4 + y4_err, color='b', alpha=0.2)
axs[3].set_xticks(x4)
axs[3].grid(True)
axs[3].set_title('(d) Contrastive loss weight.')

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/line_7.png", bbox_inches="tight")