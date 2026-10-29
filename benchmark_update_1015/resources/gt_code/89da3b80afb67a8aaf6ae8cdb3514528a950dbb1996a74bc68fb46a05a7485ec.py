import matplotlib.pyplot as plt
import numpy as np

def parametric_curve(alpha, beta, gamma, delta):
    theta = np.linspace(0, 3*np.pi, 150)
    return np.sin(theta*alpha)*np.cos(theta*beta), np.sin(theta*gamma)*np.cos(theta*delta)

fig = plt.figure(figsize=(9, 9))
main_grid = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)

for i in range(5):
    for j in range(5):
        sub_grid = main_grid[i, j].subgridspec(4, 4, wspace=0.05, hspace=0.05)
        axes = sub_grid.subplots()
        for (k, l), ax in np.ndenumerate(axes):
            x, y = parametric_curve(i + 0.5, j + 0.8, k + 0.7, l + 0.9)
            ax.plot(x, y, color='#2E86AB', linewidth=1.2, linestyle='-')
            ax.set(xticks=[], yticks=[])

for ax in fig.get_axes():
    sub_spec = ax.get_subplotspec()
    ax.spines.top.set_visible(sub_spec.is_first_row())
    ax.spines.bottom.set_visible(sub_spec.is_last_row())
    ax.spines.left.set_visible(sub_spec.is_first_col())
    ax.spines.right.set_visible(sub_spec.is_last_col())

plt.show()