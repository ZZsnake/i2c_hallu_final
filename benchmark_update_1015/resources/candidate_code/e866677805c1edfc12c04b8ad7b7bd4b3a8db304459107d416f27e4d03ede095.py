import matplotlib.pyplot as plt

# Figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

# Common properties
models = ["iTrm-All", "iTrm-En", "PatchTST", "TiDE", "TimeXer"]
colors = {
    "iTrm-All": "#A6CE39",   # light green
    "iTrm-En": "#7CA000",    # olive
    "PatchTST": "#35BFC9",   # teal
    "TiDE": "#F4A623",       # orange
    "TimeXer": "#D63F32",    # red
}
sizes = {
    "iTrm-All": 140,
    "iTrm-En": 135,
    "PatchTST": 150,
    "TiDE": 170,
    "TimeXer": 145,
}

# ---------- Top subplot: DE ----------
x_de = {
    "iTrm-All": 0.88,
    "iTrm-En": 0.82,
    "PatchTST": 0.92,
    "TiDE": 0.72,
    "TimeXer": 0.96,
}
y_de = {
    "iTrm-All": 0.096,
    "iTrm-En": 0.0965,
    "PatchTST": 0.105,
    "TiDE": 0.125,
    "TimeXer": 0.095,
}

for m in models:
    ax1.scatter(
        x_de[m],
        y_de[m],
        s=sizes[m],
        color=colors[m],
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

ax1.set_title("DE", fontsize=14)
ax1.set_xlabel("CKA Similarity", fontsize=12)
ax1.set_ylabel("MSE", fontsize=12)

ax1.set_xlim(0.6, 1.1)
ax1.set_ylim(0.09, 0.13)
ax1.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax1.set_yticks([0.09, 0.10, 0.11, 0.12, 0.13])

ax1.grid(True, linestyle=":", color="0.7")
ax1.tick_params(axis="both", labelsize=10)

# ---------- Bottom subplot: PJM ----------
x_pjm = {
    "iTrm-All": 0.89,
    "iTrm-En": 0.62,
    "PatchTST": 0.85,
    "TiDE": 0.89,
    "TimeXer": 0.92,
}
y_pjm = {
    "iTrm-All": 0.48,
    "iTrm-En": 0.48,
    "PatchTST": 0.46,
    "TiDE": 0.575,
    "TimeXer": 0.44,
}

for m in models:
    ax2.scatter(
        x_pjm[m],
        y_pjm[m],
        s=sizes[m],
        color=colors[m],
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

ax2.set_title("PJM", fontsize=14)
ax2.set_xlabel("CKA Similarity", fontsize=12)
ax2.set_ylabel("MSE", fontsize=12)

ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.40, 0.60)
ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])

ax2.grid(True, linestyle=":", color="0.7")
ax2.tick_params(axis="both", labelsize=10)

# ---------- Legend ----------
# Create dummy handles for legend using the colors
handles = []
for m in models:
    h = ax2.scatter([], [], s=sizes[m], color=colors[m], edgecolor="black", linewidth=0.7)
    handles.append(h)

fig.legend(
    handles,
    models,
    loc="lower center",
    ncol=len(models),
    frameon=False,
    bbox_to_anchor=(0.5, 0.03),
    fontsize=10,
)

plt.tight_layout(rect=[0.05, 0.07, 0.95, 0.98])
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/scatter_9.png", bbox_inches="tight")