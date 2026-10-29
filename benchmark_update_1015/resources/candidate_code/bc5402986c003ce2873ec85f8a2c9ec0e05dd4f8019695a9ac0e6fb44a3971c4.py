import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

companies_left = ['Apple', 'Samsung', 'Google']
revenue_left = [(60, 70, 10.00), (80, 88, 8.00), (70, 77, 7.00)]
market_left = [(35, 40, 5.00), (25, 28, 3.00), (20, 22, 2.00)]

companies_right = ['Microsoft', 'Huawei', 'Sony']
revenue_right = [(50, 59, 9.00), (65, 71, 6.00), (60, 65, 5.00)]
market_right = [(30, 34, 4.00), (20, 25, 5.00), (15, 18, 3.00)]


class HandlerArrowThroughLine(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        line = mpatches.Rectangle(
            (xdescent, ydescent + height/2 - 1.2),
            width,
            2.4,
            fill=True,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=0
        )
        line.set_transform(trans)

        arrow_pos_x = xdescent + width * 0.8
        p = mpatches.FancyArrow(
            arrow_pos_x, ydescent + height / 2.0,
            width*0.01, 0,
            length_includes_head=True,
            head_width=height * 1.5,
            head_length=width * 0.3,
            linewidth=0,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor()
        )
        p.set_transform(trans)
        return [line, p]


def plot_subplot(ax, companies, revenue_data, market_data, y_label_side='left'):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3)

    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.55)
    ax.yaxis.grid(False)
    for sep in [1, 2]:
        ax.axhline(sep, color='gray', linewidth=0.7, alpha=0.8, zorder=0)

    for i, company in enumerate(companies):
        rev_y = i + 0.75
        mkt_y = i + 0.25

        rev_start, rev_end, rev_val = revenue_data[i]
        ax.annotate('', xy=(rev_end, rev_y), xytext=(rev_start, rev_y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.3,
                                    shrinkA=0, shrinkB=0, mutation_scale=8),
                    zorder=2)
        ax.plot([rev_start, rev_end], [rev_y, rev_y], linestyle='None', marker='o',
                color='black', markersize=3, zorder=3)
        ax.text(rev_end - 4, rev_y + 0.08, f'{rev_val:.2f}',
                ha='center', va='bottom', fontsize=10, color='gray')

        mkt_start, mkt_end, mkt_val = market_data[i]
        ax.annotate('', xy=(mkt_end, mkt_y), xytext=(mkt_start, mkt_y),
                    arrowprops=dict(arrowstyle='->', color='darkorchid', lw=1.3,
                                    shrinkA=0, shrinkB=0, mutation_scale=8),
                    zorder=2)
        ax.plot([mkt_start, mkt_end], [mkt_y, mkt_y], linestyle='None', marker='o',
                color='black', markersize=3, zorder=3)
        ax.text(mkt_end + 3, mkt_y + 0.08, f'{mkt_val:.2f}',
                ha='center', va='bottom', fontsize=10, color='darkorchid')

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(companies)
    ax.set_xticks(range(0, 101, 10))

    if y_label_side == 'left':
        ax.yaxis.tick_left()
        ax.set_ylabel('')
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_ylabel('')

    revenue_arrow = mpatches.FancyArrow(0, 0, 1, 0, color='gray', label='Revenue Growth')
    market_arrow = mpatches.FancyArrow(0, 0, 1, 0, color='darkorchid', label='Market Share Growth')
    ax.legend(handles=[revenue_arrow, market_arrow], loc='upper center',
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, fontsize=10,
              handler_map={mpatches.FancyArrow: HandlerArrowThroughLine()})


plot_subplot(ax1, companies_left, revenue_left, market_left, y_label_side='left')
plot_subplot(ax2, companies_right, revenue_right, market_right, y_label_side='right')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/quiver_7.png", bbox_inches="tight")