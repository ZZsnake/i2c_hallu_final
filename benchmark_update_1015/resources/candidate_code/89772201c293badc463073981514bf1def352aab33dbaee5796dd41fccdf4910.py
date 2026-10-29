import matplotlib.pyplot as plt
import numpy as np

# Set figure size
fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=False, constrained_layout=True)

# Data for subplot (a) Positive bound
x_a = np.array([0.70, 0.75, 0.80, 0.85, 0.90])
y_a = np.array([76.0, 78.0, 80.0, 79.0, 77.0])
y_a_upper = np.array([79.0, 80.0, 84.0, 81.0, 80.0])
y_a_lower = np.array([73.0, 76.0, 76.0, 77.0, 74.0])
axes[0].plot(x_a, y_a, 'o-', color='blue', linewidth=2, markersize=6)
axes[0].fill_between(x_a, y_a_lower, y_a_upper, color='lightblue', alpha=0.5)
axes[0].set_xlim(0.68, 0.92)
axes[0].set_ylim(70, 84)
axes[0].set_xticks(x_a)
axes[0].set_yticks(np.arange(70, 85, 2))
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_xlabel('Positive bound.')
axes[0].set_ylabel('Value')
axes[0].set_title('(a) Positive bound.')

# Data for subplot (b) Negative bound
x_b = np.array([0.1, 0.2, 0.3, 0.4])
y_b = np.array([74.0, 78.0, 76.0, 72.0])
y_b_upper = np.array([77.0, 80.0, 78.0, 74.0])
y_b_lower = np.array([71.0, 76.0, 74.0, 70.0])
axes[1].plot(x_b, y_b, 'o-', color='blue', linewidth=2, markersize=6)
axes[1].fill_between(x_b, y_b_lower, y_b_upper, color='lightblue', alpha=0.5)
axes[1].set_xlim(0.08, 0.42)
axes[1].set_ylim(70, 80)
axes[1].set_xticks(x_b)
axes[1].set_yticks(np.arange(70, 81, 2))
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_xlabel('Negative bound.')
axes[1].set_title('(b) Negative bound.')

# Data for subplot (d) Contrastive loss weight
x_d = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y_d = np.array([77.0, 78.0, 70.0, 65.0, 60.0, 55.0])
y_d_upper = np.array([80.0, 80.0, 73.0, 68.0, 63.0, 58.0])
y_d_lower = np.array([74.0, 76.0, 67.0, 62.0, 57.0, 52.0])
axes[2].plot(x_d, y_d, 'o-', color='blue', linewidth=2, markersize=6)
axes[2].fill_between(x_d, y_d_lower, y_d_upper, color='lightblue', alpha=0.5)
axes[2].set_xlim(-0.05, 1.05)
axes[2].set_ylim(50, 80)
axes[2].set_xticks(x_d)
axes[2].set_yticks(np.arange(50, 81, 5))
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].set_xlabel('Contrastive loss weight.')
axes[2].set_title('(d) Contrastive loss weight.')

# Data for subplot (c) Fuzzy coefficient
x_c = np.array([400, 600, 800, 1000, 1200])
y_c = np.array([80.0, 82.5, 77.5, 75.0, 72.5])
y_c_upper = np.array([82.5, 85.0, 80.0, 77.5, 75.0])
y_c_lower = np.array([77.5, 80.0, 75.0, 72.5, 70.0])
axes[3].plot(x_c, y_c, 'o-', color='blue', linewidth=2, markersize=6)
axes[3].fill_between(x_c, y_c_lower, y_c_upper, color='lightblue', alpha=0.5)
axes[3].set_xlim(350, 1250)
axes[3].set_ylim(67.5, 85.0)
axes[3].set_xticks(x_c)
axes[3].set_yticks(np.arange(67.5, 85.1, 2.5))
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].set_xlabel('Fuzzy coefficient.')
axes[3].set_title('(c) Fuzzy coefficient.')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/line_7.png", bbox_inches="tight")