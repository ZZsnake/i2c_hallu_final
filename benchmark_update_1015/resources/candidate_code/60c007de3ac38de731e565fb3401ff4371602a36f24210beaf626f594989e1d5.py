import matplotlib.pyplot as plt
import numpy as np

# Set the figure size
plt.figure(figsize=(6, 4))

# Generate dummy data for box plots
np.random.seed(42)
data1 = np.random.normal(20.5, 1.5, 100)  # Llama-70b (T2TT)
data2 = np.random.normal(19.2, 1.0, 100)  # Llama-70b (S2TT)

# Create box plots
positions = [1, 2]
plt.boxplot([data1, data2], positions=positions, patch_artist=True,
            boxprops=dict(facecolor='white', color='black'),
            medianprops=dict(color='orange', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Customize x-axis labels
plt.xticks(positions, ['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'])

# Set y-axis label
plt.ylabel('BLEU')

# Set y-axis limits
plt.ylim(16, 22)

# Add a dashed line and annotation for the 1.0 BLEU difference
plt.axhline(y=20.5, color='gray', linestyle='--', linewidth=1)
plt.annotate('1.0 BLEU', xy=(1.5, 20.5), xytext=(1.5, 21.5),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, ha='center')

# Customize plot appearance
plt.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/box_11.png", bbox_inches="tight")