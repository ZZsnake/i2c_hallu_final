import matplotlib.pyplot as plt
import numpy as np

labels = ['rose', '3D-bar', 'candlestick', 'line_num', 'heatmap', 'line', 'treemap', 'box', 'bar_num', 'histogram', 
          'funnel', 'pie', 'area', 'radar', 'bubble', 'multi-axes']
num_vars = len(labels)

# Dummy data matching approximate values from the image
qwen_vl = [4.8, 4.5, 4.0, 3.5, 3.8, 4.0, 4.5, 4.2, 3.0, 2.5, 3.5, 4.0, 4.5, 3.8, 3.5, 4.5]
sphinx_v2 = [3.5, 4.8, 4.5, 4.2, 3.5, 3.8, 4.2, 4.5, 4.0, 3.0, 2.5, 3.5, 4.0, 4.5, 4.0, 4.2]
chart_llama = [2.5, 3.0, 3.5, 3.2, 3.0, 3.5, 3.8, 3.5, 3.2, 2.8, 2.5, 3.0, 3.5, 3.2, 3.8, 3.5]

# Close the loop for radar chart
qwen_vl += qwen_vl[:1]
sphinx_v2 += sphinx_v2[:1]
chart_llama += chart_llama[:1]

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')

# Plot each group with specific style
ax.plot(angles, qwen_vl, color='red', linewidth=2, linestyle='solid', marker='o', markersize=6, label='QWen-VL')
ax.fill(angles, qwen_vl, color='red', alpha=0.25)

ax.plot(angles, sphinx_v2, color='steelblue', linewidth=2, linestyle='solid', marker='s', markersize=6, label='SPHINX-V2')
ax.fill(angles, sphinx_v2, color='steelblue', alpha=0.15)

ax.plot(angles, chart_llama, color='orange', linewidth=2, linestyle='solid', marker='D', markersize=6, label='ChartLlama')
ax.fill(angles, chart_llama, color='orange', alpha=0.1)

ax.set_ylim(1, 5)

# Configure radial axis
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], color='gray', size=9)
ax.yaxis.grid(True, linestyle='--', color='0.85', linewidth=0.7)

# Configure angular axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=9, color='black', ha='center')
ax.xaxis.grid(True, linestyle='--', color='0.85', linewidth=0.7)

# Add legend and clean up layout
plt.legend(loc='lower left', bbox_to_anchor=(0.1, -0.1), ncol=3, frameon=False, fontsize=10)
ax.spines['polar'].set_visible(False)
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/radar_7.png", bbox_inches="tight")