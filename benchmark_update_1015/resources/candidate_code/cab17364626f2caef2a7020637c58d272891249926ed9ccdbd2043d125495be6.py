import numpy as np
import matplotlib.pyplot as plt

# Data
labels = ['line', 'heatmap', 'line_num', 'candlestick', '3D-bar', 'rose', 'multi-axes', 'bubble', 'radar', 'area', 'pie', 'funnel', 'histogram', 'bar_num', 'box', 'treemap']
num_vars = len(labels)

# Performance data for each model
QWen_VL = [3.0, 3.3, 4.2, 4.0, 4.8, 5.0, 4.0, 3.2, 4.0, 4.4, 3.5, 3.8, 4.5, 4.3, 3.6, 4.0]
SPHINX_V2 = [3.5, 3.4, 3.6, 3.5, 3.7, 3.2, 3.5, 4.0, 4.5, 3.8, 4.0, 2.8, 3.0, 4.1, 3.0, 3.0]
ChartLlama = [2.0, 2.3, 2.6, 2.7, 2.5, 2.6, 2.5, 2.2, 1.4, 2.6, 1.5, 1.4, 2.2, 2.4, 1.5, 1.4]

# Compute angle for each label
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Close the loop
QWen_VL += QWen_VL[:1]
SPHINX_V2 += SPHINX_V2[:1]
ChartLlama += ChartLlama[:1]

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, QWen_VL, color='darkred', alpha=0.25)
ax.fill(angles, SPHINX_V2, color='steelblue', alpha=0.25)
ax.fill(angles, ChartLlama, color='orange', alpha=0.25)

ax.plot(angles, QWen_VL, color='darkred', linewidth=2, marker='o', markersize=6, label='QWen-VL')
ax.plot(angles, SPHINX_V2, color='steelblue', linewidth=2, linestyle='--', marker='s', markersize=6, label='SPHINX-V2')
ax.plot(angles, ChartLlama, color='orange', linewidth=2, linestyle=':', marker='D', markersize=6, label='ChartLlama')

# Labels
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Legend
plt.legend(loc='lower left', bbox_to_anchor=(0.1, -0.2), fontsize='small')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/radar_7.png", bbox_inches="tight")
