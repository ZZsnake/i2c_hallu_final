import matplotlib.pyplot as plt

# Set figure size
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Define company names and data for left plot
left_companies = ['Google', 'Samsung', 'Apple']
left_revenue_start = [70, 80, 60]
left_revenue_end = [77, 88, 70]
left_market_start = [20, 25, 35]
left_market_end = [22, 28, 40]
left_revenue_labels = [7.00, 8.00, 10.00]
left_market_labels = [2.00, 3.00, 5.00]

# Define company names and data for right plot
right_companies = ['Sony', 'Huawei', 'Microsoft']
right_revenue_start = [60, 65, 50]
right_revenue_end = [65, 71, 59]
right_market_start = [15, 20, 30]
right_market_end = [18, 25, 34]
right_revenue_labels = [5.00, 6.00, 9.00]
right_market_labels = [3.00, 5.00, 4.00]

# Common styling
colors = {'revenue': 'gray', 'market': 'purple'}
arrow_props = {'arrowstyle': '-|>', 'lw': 2}

# Plot left subplot
ax1 = axes[0]
ax1.set_yticks(range(len(left_companies)))
ax1.set_yticklabels(left_companies)
ax1.set_ylim(len(left_companies) - 0.5, -0.5)
ax1.set_xlim(0, 100)
ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
ax1.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')
ax1.hlines([0.5, 1.5], 0, 100, colors='black', linewidth=0.5, alpha=0.7)

for i, company in enumerate(left_companies):
    revenue_y = i - 0.25
    market_y = i + 0.25

    # Revenue Growth (gray arrow)
    ax1.annotate("", xy=(left_revenue_end[i], revenue_y), xytext=(left_revenue_start[i], revenue_y),
                 arrowprops={'arrowstyle': '->', 'color': colors['revenue'], 'lw': 1})
    ax1.plot(left_revenue_start[i], revenue_y, 'ko', markersize=4)
    ax1.plot(left_revenue_end[i], revenue_y, 'ko', markersize=4)
    ax1.text(left_revenue_end[i], revenue_y - 0.05, f"{left_revenue_labels[i]:.2f}", 
             color=colors['revenue'], fontsize=9, ha='right', va='bottom')

    # Market Share Growth (purple arrow)
    ax1.annotate("", xy=(left_market_end[i], market_y), xytext=(left_market_start[i], market_y),
                 arrowprops={'arrowstyle': '->', 'color': colors['market'], 'lw': 1})
    ax1.plot(left_market_start[i], market_y, 'ko', markersize=4)
    ax1.plot(left_market_end[i], market_y, 'ko', markersize=4)
    ax1.text(left_market_end[i] + 1, market_y - 0.05, f"{left_market_labels[i]:.2f}", 
             color=colors['market'], fontsize=9, ha='left', va='bottom')

# Plot right subplot
ax2 = axes[1]
ax2.set_yticks(range(len(right_companies)))
ax2.set_yticklabels(right_companies)
ax2.yaxis.set_ticks_position('right')
ax2.tick_params(axis='y', labelleft=False, labelright=True)
ax2.set_ylim(len(right_companies) - 0.5, -0.5)
ax2.set_xlim(0, 100)
ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
ax2.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')
ax2.hlines([0.5, 1.5], 0, 100, colors='black', linewidth=0.5, alpha=0.7)

for i, company in enumerate(right_companies):
    revenue_y = i - 0.25
    market_y = i + 0.25

    # Revenue Growth (gray arrow)
    ax2.annotate("", xy=(right_revenue_end[i], revenue_y), xytext=(right_revenue_start[i], revenue_y),
                 arrowprops={'arrowstyle': '->', 'color': colors['revenue'], 'lw': 1})
    ax2.plot(right_revenue_start[i], revenue_y, 'ko', markersize=4)
    ax2.plot(right_revenue_end[i], revenue_y, 'ko', markersize=4)
    ax2.text(right_revenue_end[i], revenue_y - 0.05, f"{right_revenue_labels[i]:.2f}", 
             color=colors['revenue'], fontsize=9, ha='right', va='bottom')

    # Market Share Growth (purple arrow)
    ax2.annotate("", xy=(right_market_end[i], market_y), xytext=(right_market_start[i], market_y),
                 arrowprops={'arrowstyle': '->', 'color': colors['market'], 'lw': 1})
    ax2.plot(right_market_start[i], market_y, 'ko', markersize=4)
    ax2.plot(right_market_end[i], market_y, 'ko', markersize=4)
    ax2.text(right_market_end[i] + 1, market_y - 0.05, f"{right_market_labels[i]:.2f}", 
             color=colors['market'], fontsize=9, ha='left', va='bottom')

# Add legends
legend_elements = [
    plt.Line2D([0], [0], color=colors['revenue'], lw=2, marker='>', markersize=10, label='Revenue Growth'),
    plt.Line2D([0], [0], color=colors['market'], lw=2, marker='>', markersize=10, label='Market Share Growth')
]

ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)
ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.4, -0.1), ncol=2, frameon=True)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/quiver_7.png", bbox_inches="tight")
