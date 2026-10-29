import matplotlib.pyplot as plt
import numpy as np

# 1. Set up the figure and axes
fig, ax = plt.subplots(figsize=(9, 7))

# 2. Generate data based on the function
# Use a large number of points to make the markers appear as a continuous line
t = np.linspace(0, 5, 100)
v = 0.5 * t**2 + 2 * t - 1

# 3. Plot the data with specific styling
ax.plot(t, v,
        marker='s',              # Square markers
        markersize=5,            # Small marker size
        markerfacecolor='red',      # Bright red marker face
        color='#9A2A2A',         # Dark red color for the marker face
        markeredgecolor='black', # Black edge for the markers
        markeredgewidth=0.5,     # Thin edge width
        linestyle='-',           # Connect markers with a line
        zorder=2)                # Ensure plot is above the grid

# 4. Set the title and labels
# Use LaTeX for the superscript in the title
ax.set_title(r'Quadratic Velocity Function: v(t) = 0.5t$^2$ + 2t - 1',
             fontweight='bold',
             fontsize=14)
ax.set_xlabel("Time (s)", fontsize=12, fontfamily='serif')
ax.set_ylabel("Velocity (m/s)", fontsize=12, fontfamily='serif')

# 5. Configure the grid
ax.grid(True, linestyle=':', color='gray', zorder=0)

# 6. Set axis limits and ticks to match the original image
ax.set_xlim(-0.2, 5.2)
ax.set_ylim(-2, 23)
ax.set_xticks(np.arange(0, 6, 1))
ax.set_yticks(np.arange(0, 21, 5))
plt.setp(ax.get_xticklabels(), rotation=15)
plt.setp(ax.get_yticklabels(), rotation=60)

# 7. Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/3e3fa84e1f6e40f1ac62373f79a0ff2b.png", bbox_inches="tight")
