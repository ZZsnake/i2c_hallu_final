import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot data
companies_left = ['Apple', 'Samsung', 'Google']
revenue_left = [(60, 70, 10.00), (80, 90, 8.00), (70, 78, 7.00)]
market_left = [(32, 38, 5.00), (25, 28, 3.00), (20, 22, 2.00)]

# Right subplot data
companies_right = ['Microsoft', 'Huawei', 'Sony']
revenue_right = [(52, 61, 9.00), (68, 75, 6.00), (62, 68, 5.00)]
market_right = [(30, 34, 4.00), (22, 27, 5.00), (20, 23, 3.00)]

def plot_subplot(ax, companies, revenue_data, market_data, y_label_side='left'):
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    
    for i, company in enumerate(companies):
        y_pos = i
        
        # Revenue Growth arrow
        rev_start, rev_end, rev_val = revenue_data[i]
        ax.plot([rev_start, rev_end], [y_pos, y_pos], 'o-', color='gray', 
                markersize=4, linewidth=1.5, zorder=2)
        ax.text((rev_start + rev_end) / 2, y_pos + 0.15, f'{rev_val:.2f}', 
                ha='center', va='bottom', fontsize=9, color='gray')
        
        # Market Share Growth arrow
        mkt_start, mkt_end, mkt_val = market_data[i]
        ax.plot([mkt_start, mkt_end], [y_pos, y_pos], color='purple', 
                marker='>', markersize=5, linewidth=1.5, zorder=3)
        ax.plot(mkt_start, y_pos, marker='o', color='purple', markersize=4, zorder=3)
        ax.text((mkt_start + mkt_end) / 2, y_pos + 0.15, f'{mkt_val:.2f}', 
                ha='center', va='bottom', fontsize=9, color='purple')
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(companies)
    ax.set_xticks(range(0, 101, 10))
    
    if y_label_side == 'left':
        ax.yaxis.tick_left()
        ax.set_ylabel('')
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_ylabel('')
    
    # Legend
    revenue_line = Line2D([0], [0], color='gray', marker='o', markersize=4, 
                          linewidth=1.5, label='Revenue Growth')
    market_line = Line2D([0], [0], color='purple', marker='>', markersize=5, 
                         linewidth=1.5, label='Market Share Growth')
    ax.legend(handles=[revenue_line, market_line], loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)

# Plot left subplot
plot_subplot(ax1, companies_left, revenue_left, market_left, y_label_side='left')

# Plot right subplot
plot_subplot(ax2, companies_right, revenue_right, market_right, y_label_side='right')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/quiver_7.png", bbox_inches="tight")