import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 1. Data and Categories
categories = ['Creativity', 'Longevity', 'Sillage', 'Packaging', 'Popularity', 'Value', 'Projection', 'Versatility']
N = len(categories)

# Data extracted from the charts
chanel_data = [8, 6, 8, 7, 5, 4, 6, 7]
dior_data = [9, 7, 9, 6, 8, 5, 8, 7]
gucci_data = [8, 7, 7, 8, 9, 7, 8, 9]

# Close the loop for plotting
chanel_data += chanel_data[:1]
dior_data += dior_data[:1]
gucci_data += gucci_data[:1]

all_data = [chanel_data, dior_data, gucci_data]
brand_names = ['Chanel', 'Dior', 'Gucci']

# Visual styles
colors = [
    {'fill': '#fef08a', 'marker': '#facc15', 'line': '#3b82b1'},  # Yellow for Chanel
    {'fill': '#e5e7eb', 'marker': '#9ca3af', 'line': '#3b82b1'},  # Gray for Dior
    {'fill': '#dcfce7', 'marker': '#22c55e', 'line': '#3b82b1'}   # Green for Gucci
]

# Angles for the radar chart
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 2. Create Figure and Subplots
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Perfume Brand Comparison', fontsize=20, y=1.05)

for i, data in enumerate(all_data):
    ax = fig.add_subplot(1, 3, i + 1, polar=True)

    # Set background color for each subplot's polar area
    ax.set_facecolor(colors[i]['fill'])
    ax.patch.set_alpha(0.4) # Set alpha for the fill color

    # Plot data
    ax.plot(angles, data, color=colors[i]['line'], linewidth=2, linestyle='solid')
    ax.scatter(angles, data, color=colors[i]['marker'], s=60, zorder=5)

    # Set axes and grid
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.tick_params(axis='x', pad=15)

    # Customize grid lines
    ax.grid(color='#cccccc', linestyle='-', linewidth=1)
    
    # Hide default y-axis labels and set custom ones
    ax.set_yticklabels([])
    ax.set_rlabel_position(45) # Position for the labels (angle in degrees)
    
    # Manually add y-tick labels on the 'Longevity' axis
    for val in [1, 3, 5, 7, 9]:
        ax.text(np.pi/4, val, str(val), color='#1e3a8a', ha='center', va='center', fontsize=10)

# 3. Create Legend
legend_patches = [
    mpatches.Patch(color=colors[0]['fill'], alpha=0.8, label='Chanel'),
    mpatches.Patch(color=colors[1]['fill'], alpha=0.8, label='Dior'),
    mpatches.Patch(color=colors[2]['fill'], alpha=0.8, label='Gucci')
]

# The legend in the original image is unusual. This creates a more standard, informative legend.
# To more closely match the original's text-based legend, one could use:
# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], color='#3b82b1', lw=2)] * 3
# fig.legend(custom_lines, brand_names, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=12)

# Using a standard patch legend for better clarity
fig.legend(handles=legend_patches,
           labels=brand_names,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=3,
           frameon=False,
           fontsize=12)

# 4. Adjust Layout
plt.subplots_adjust(wspace=0.5, top=0.85, bottom=0.1)
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/radar_20.png", bbox_inches="tight")