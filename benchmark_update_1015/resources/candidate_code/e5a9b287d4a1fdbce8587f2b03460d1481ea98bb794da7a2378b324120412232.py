import matplotlib.pyplot as plt

# Figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

# Common properties
models = ["iTrm-All", "iTrm-En", "PatchTST", "TiDE", "TimeXer"]
colors = {
    "iTrm-All": "#99CC58",   # light green
    "iTrm-En": "#99CC58",    # light green
    "PatchTST": "#55BDBD",   # muted teal
    "TiDE": "#E6954A",       # muted orange
    "TimeXer": "#D63F32",    # red
}
sizes = {
    "iTrm-All": 200,
    "iTrm-En": 200,
    "PatchTST": 200,
    "TiDE": 200,
    "TimeXer": 200,
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
    "iTrm-En": 0.096,
    "PatchTST": 0.105,
    "TiDE": 0.125,
    "TimeXer": 0.095,
}

for m in models:
    if m == "iTrm-All":
        ax1.scatter(
            x_de[m],
            y_de[m],
            s=sizes[m],
            facecolors="none",
            edgecolors=colors[m],
            linewidth=3.2,
            zorder=3,
        )
    else:
        ax1.scatter(
            x_de[m],
            y_de[m],
            s=sizes[m],
            color=colors[m],
            edgecolors="none",
            linewidth=0,
            zorder=3,
        )

ax1.set_title("DE", fontsize=12)
ax1.set_xlabel("CKA Similarity", fontsize=10)
ax1.set_ylabel("MSE", fontsize=10)

ax1.set_xlim(0.6, 1.1)
ax1.set_ylim(0.09, 0.13)
ax1.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax1.set_yticks([0.09, 0.10, 0.11, 0.12, 0.13])

ax1.grid(True, linestyle="--", color="black", linewidth=0.5)
ax1.tick_params(axis="both", labelsize=10)

# ---------- Bottom subplot: PJM ----------
x_pjm = {
    "iTrm-All": 0.88,
    "iTrm-En": 0.62,
    "PatchTST": 0.86,
    "TiDE": 0.89,
    "TimeXer": 0.92,
}
y_pjm = {
    "iTrm-All": 0.48,
    "iTrm-En": 0.48,
    "PatchTST": 0.46,
    "TiDE": 0.58,
    "TimeXer": 0.44,
}

for m in models:
    if m == "iTrm-All":
        ax2.scatter(
            x_pjm[m],
            y_pjm[m],
            s=sizes[m],
            facecolors="none",
            edgecolors=colors[m],
            linewidth=3.2,
            zorder=3,
        )
    else:
        ax2.scatter(
            x_pjm[m],
            y_pjm[m],
            s=sizes[m],
            color=colors[m],
            edgecolors="none",
            linewidth=0,
            zorder=3,
        )

ax2.set_title("PJM", fontsize=12)
ax2.set_xlabel("CKA Similarity", fontsize=10)
ax2.set_ylabel("MSE", fontsize=10)

ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.40, 0.60)
ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])

ax2.grid(True, linestyle="--", color="black", linewidth=0.5)
ax2.tick_params(axis="both", labelsize=10)

# ---------- Legend ----------
# Create dummy handles for legend using the colors
handles = []
for m in models:
    if m == "iTrm-All":
        h = ax2.scatter([], [], s=sizes[m], facecolors="none", edgecolors=colors[m], linewidth=3.2)
    else:
        h = ax2.scatter([], [], s=80, color=colors[m], edgecolors="none", linewidth=0)
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

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/scatter_9.png", bbox_inches="tight")
