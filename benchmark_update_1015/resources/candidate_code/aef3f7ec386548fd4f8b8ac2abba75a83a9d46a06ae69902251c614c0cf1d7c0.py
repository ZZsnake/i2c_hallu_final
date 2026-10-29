import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig_handle, (top_axes, bottom_axes) = plt.subplots(
    2, 1, figsize=(10, 14), subplot_kw={'projection': '3d'})

# Generate analytical geometry test dataset
X_coords, Y_coords, Z_coords = axes3d.get_test_data(0.08)

# Top subplot: Wireframes with constant column spacing
top_axes.plot_wireframe(X_coords, Y_coords, Z_coords, 
                     rstride=12, cstride=0, 
                     color='#2E86AB', linewidth=1.2)
top_axes.set_title("Constant Column Parameterization", fontsize=14, fontweight='bold')

# Bottom subplot: Wireframes with constant row spacing  
bottom_axes.plot_wireframe(X_coords, Y_coords, Z_coords,
                        rstride=0, cstride=12,
                        color='#A23B72', linewidth=1.2)
bottom_axes.set_title("Constant Row Parameterization", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
