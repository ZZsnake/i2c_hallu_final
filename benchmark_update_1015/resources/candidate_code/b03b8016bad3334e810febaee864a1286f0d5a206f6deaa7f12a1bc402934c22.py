import matplotlib.pyplot as plt
import numpy as np

# Define the categories (axes)
categories = ['Creativity', 'Longevity', 'Sillage', 'Packaging', 'Popularity', 'Value', 'Projection', 'Versatility']

# Define data for each brand (scores from 1 to 9)
chanel_data = [8, 7, 8, 8, 7, 6, 6, 6]
dior_data = [8, 7, 6, 7, 6, 5, 7, 6]
gucci_data = [7, 7, 5, 7, 7, 7, 7, 8]

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
colors = ['#FFD700', '#D3D3D3', '#8FBC8F']  # Yellow, Gray, Light Green
labels = ['Chanel', 'Dior', 'Gucci']

# Plot each brand in its own subplot
for i, (ax, data, color, label) in enumerate(zip(axs, [chanel_data, dior_data, gucci_data], colors, labels)):
    # Plot the radar chart
    ax.plot(angles, data, color='navy', linewidth=2, label=label)
    ax.fill(angles, data, color=color, alpha=0.3)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color='darkblue')
    
    # Set y-axis ticks
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_yticklabels([1, 3, 5, 7, 9], fontsize=10, color='darkblue')
    
    # Set grid
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Set title for each subplot
    ax.set_title(label, fontsize=12, pad=20, color='darkblue')

# Add main title
fig.suptitle('Perfume Brand Comparison', fontsize=16, fontweight='bold', color='darkblue', y=0.95)

# Add legend below the middle subplot
handles = [plt.Line2D([0], [0], color='navy', lw=2, label=label) for label in labels]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=False, fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/radar_20.png", bbox_inches="tight")