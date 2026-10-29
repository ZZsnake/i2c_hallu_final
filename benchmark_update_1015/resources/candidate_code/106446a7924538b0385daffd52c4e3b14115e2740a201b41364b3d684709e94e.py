import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on visual trends
data = {
    'Llama-70b (T2TT)': np.random.normal(20.5, 1, 100),
    'Llama-70b (S2TT)': np.random.normal(19, 1, 100)
}

# Figure setup
fig, ax = plt.subplots(figsize=(6, 4))

# Box plot creation
bp = ax.boxplot([data['Llama-70b (T2TT)'], data['Llama-70b (S2TT)']], patch_artist=True)

# Customize box plot appearance
for box in bp['boxes']:
    box.set(facecolor='white', edgecolor='black')
for median in bp['medians']:
    median.set(color='orange')

# Add dashed line for reference
ax.axhline(y=20, color='gray', linestyle='--', label='1.0 BLEU')

# Axis labels and ticks
ax.set_xticklabels(['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'])
ax.set_ylabel('BLEU')
ax.set_ylim(16, 22)

# Legend
ax.legend()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/box_11.png", bbox_inches="tight")