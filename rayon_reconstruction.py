import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

np.random.seed(42)
n_points = 30
t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
noise = 0.05
points = np.column_stack((np.cos(t) + np.random.uniform(-noise, noise, n_points),
                          np.sin(t) + np.random.uniform(-noise, noise, n_points)))

radii = [0.1, 0.4, 1.2] 
titles = ["Rayon r=0.1 (Trop petit)\nTopologie = Poussière", 
          "Rayon r=0.4 (Reconstruction)\nTopologie = Cercle (Isotope à M)", 
          "Rayon r=1.2 (Trop grand)\nTopologie = Disque (Trou bouché)"]

fig, axes = plt.subplots(1, 3, figsize=(25, 10))

for ax, r, title in zip(axes, radii, titles):
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    for p in points:
        circle = Circle(p, r, color='dodgerblue', alpha=0.4, edgecolor='none')
        ax.add_patch(circle)
        
    ax.scatter(points[:, 0], points[:, 1], color='black', s=10, zorder=10)
    
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()