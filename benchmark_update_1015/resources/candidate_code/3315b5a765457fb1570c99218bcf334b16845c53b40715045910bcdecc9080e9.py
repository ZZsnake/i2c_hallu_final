import matplotlib.pyplot as plt
import numpy as np

# 1. Data Preparation
# Data for the APTQ line plot
aptq_x = np.array([40, 50, 60, 70, 80, 90, 100])
aptq_y = np.array([7.5, 6.5, 6.0, 5.75, 5.6, 5.5, 5.5])
aptq_annotations = ["2.91 bit", "3.11 bit", "3.32 bit", "3.53 bit", "3.63 bit", "3.74 bit", "3.94 bit"]

# Data for the horizontal baseline models
baselines = {
    "LLaMa-7B (FP16): 5.22": {'y': 5.22, 'color': '#9932CC', 'style': '--'},
    "OWQ-4bit: 5.56": {'y': 5.56, 'color': '#BDB76B', 'style': '--'},
    "GPTQ-4bit: 5.62": {'y': 5.62, 'color': '#2E8B57', 'style': '--'},
    "LLM-QAT-4bit: 7.4": {'y': 7.4, 'color': '#B22222', 'style': '--'}
}

# 2. Plotting
# Set figure dimensions
fig, ax = plt.subplots(figsize=(5, 4))

# Plot the horizontal baseline lines
for label, props in baselines.items():
    ax.axhline(y=props['y'], color=props['color'], linestyle=props['style'], label=label)

# Plot the main APTQ line
ax.plot(aptq_x, aptq_y, marker='o', color='#00008B', label='APTQ')

# 3. Annotations
# Add text annotations for each point on the APTQ line
annotation_offsets = [(-15, 5), (3, 5), (3, 5), (3, 5), (3, 5), (3, 5), (3, 5)]
for i, txt in enumerate(aptq_annotations):
    ax.annotate(txt, (aptq_x[i], aptq_y[i]), textcoords="offset points", xytext=annotation_offsets[i], ha='center')

# 4. Styling and Labels
# Set axis labels
ax.set_xlabel("Ratio of 4-bit Utilization (%)")
ax.set_ylabel("Perplexity (PPL)")

# Set axis limits and ticks
ax.set_xlim(35, 105)
ax.set_ylim(4.9, 7.8)
ax.set_xticks(np.arange(40, 101, 10))
ax.set_yticks(np.arange(5.0, 7.6, 0.5))

# Add grid
ax.grid(True, which='both', linestyle='-', color='lightgray')

# Add legend
# Reordering handles and labels to match the image
handles, labels = ax.get_legend_handles_labels()
order = [4, 0, 1, 2, 3]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center right', bbox_to_anchor=(1, 0.7))

# Ensure tight layout
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/line_22.png", bbox_inches="tight")