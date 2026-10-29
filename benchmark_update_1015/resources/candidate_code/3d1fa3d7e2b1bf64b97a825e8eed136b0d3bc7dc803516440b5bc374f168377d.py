import matplotlib.pyplot as plt
import numpy as np

# Define the categories (axes)
categories = ['Sillage', 'Longevity', 'Creativity', 'Versatility', 'Projection', 'Value', 'Popularity', 'Packaging']

# Define data for each brand (scores from 0 to 10)
chanel_data = [8, 6, 7, 5, 6, 7, 8, 9]
dior_data = [7, 7.5, 8, 6, 7, 5, 8.5, 7]
gucci_data = [5, 7, 6.5, 8, 7, 6, 7, 7.5]

# Convert to numpy arrays for easier manipulation
chanel_data = np.array(chanel_data)
dior_data = np.array(dior_data)
gucci_data = np.array(gucci_data)

# Add the first value to the end to close the polygon
chanel_data = np.concatenate((chanel_data, [chanel_data[0]]))
dior_data = np.concatenate((dior_data, [dior_data[0]]))
gucci_data = np.concatenate((gucci_data, [gucci_data[0]]))

# Define angles for the radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Set up the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

# Define colors for each brand
colors = ['#FFD700', '#D3D3D3', 'green']  # Yellow, Gray, Light Green
labels = ['Chanel', 'Dior', 'Gucci']

# Plot each brand in its own subplot
for i, (ax, data, color, label) in enumerate(zip(axs, [chanel_data, dior_data, gucci_data], colors, labels)):
    # Plot the radar chart
    ax.plot(angles, data, color='#1f77b4', linewidth=2, marker='o', markersize=7,
            markerfacecolor=color, markeredgecolor=color, label=label)
    ax.fill(angles, data, color=color, alpha=0.3)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='normal', color='darkblue')
    
    # Set y-axis ticks
    ax.set_ylim(0, 10)
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_yticklabels([1, 3, 5, 7, 9], fontsize=10, color='darkblue')
    
    # Set grid
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.8)

# Add main title
fig.suptitle('Perfume Brand Comparison', fontsize=19, fontweight='normal', color='black', y=0.95)

# Add legend below the middle subplot
handles = [plt.Line2D([0], [0], color='#1f77b4', lw=2, label=label) for label in labels]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.1, 1, 0.9])

# Show the plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/radar_20.png", bbox_inches="tight")
