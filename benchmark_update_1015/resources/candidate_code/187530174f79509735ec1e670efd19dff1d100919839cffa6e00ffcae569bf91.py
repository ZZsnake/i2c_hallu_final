import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Central horizontal line
ax.hlines(0.5, 0, 0.8, colors='black', linewidth=1)

# Funnel shape
funnel = patches.Polygon([[0.8, 0.4], [0.8, 0.6], [1.0, 0.5]], 
                         closed=True, facecolor='red', edgecolor='none')
ax.add_patch(funnel)
ax.text(1.02, 0.5, 'ISSUE', va='center', ha='left', fontsize=12, color='red')

# Stage configurations
stages = [
    {'x': 0.1, 'y': 0.7, 'name': 'PLANNING', 'issues': 
        ['Resource allocation', 'Budget constraints', 'Timeline issues', 'Scope changes'], 'dir': 'top'},
    {'x': 0.35, 'y': 0.7, 'name': 'DEVELOPMENT', 'issues': 
        ['Integration problems', 'Coding errors', 'Testing gaps'], 'dir': 'top'},
    {'x': 0.6, 'y': 0.7, 'name': 'MONITORING', 'issues': 
        ['Error tracking', 'Performance metrics', 'User feedback'], 'dir': 'top'},
    {'x': 0.1, 'y': 0.3, 'name': 'DESIGN', 'issues': 
        ['Aesthetics', 'Technical limitations', 'User experience'], 'dir': 'bottom'},
    {'x': 0.35, 'y': 0.3, 'name': 'DEPLOYMENT', 'issues': 
        ['Security concerns', 'Server configuration', 'Database setup'], 'dir': 'bottom'},
    {'x': 0.6, 'y': 0.3, 'name': 'MAINTENANCE', 'issues': 
        ['Technical debt', 'Bug fixes', 'Updates'], 'dir': 'bottom'}
]

# Draw stages and annotations
for stage in stages:
    # Stage label
    ax.text(stage['x'], stage['y'], stage['name'], 
            ha='center', va='center', color='white',
            bbox=dict(facecolor='red', edgecolor='none', boxstyle='square,pad=0.5'))
    
    # Issue annotations
    for i, issue in enumerate(stage['issues']):
        offset = 0.06 + i * 0.04
        text_y = stage['y'] - offset if stage['dir'] == 'top' else stage['y'] + offset
        ax.annotate(issue, xy=(stage['x'], 0.5), 
                    xytext=(stage['x'] + 0.03, text_y),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    ha='left', va='center')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")