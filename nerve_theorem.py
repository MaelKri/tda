import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
radius_circle = 1.0
points = np.column_stack((radius_circle * np.cos(angles), radius_circle * np.sin(angles)))

r_cover = 0.6

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.scatter(points[:, 0], points[:, 1], color='black', zorder=10, label='Points de données')

for i, p in enumerate(points):
    circle = Circle(p, r_cover, color='skyblue', alpha=0.4, edgecolor='blue')
    ax.add_patch(circle)
    ax.text(p[0]*1.2, p[1]*1.2, f"$U_{i}$", fontsize=12, ha='center')

nerve_edges = []
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        dist = np.linalg.norm(points[i] - points[j])
        if dist < 2 * r_cover:
            nerve_edges.append((points[i], points[j]))
            ax.plot([points[i][0], points[j][0]], 
                    [points[i][1], points[j][1]], 
                    color='red', linewidth=2, linestyle='--')

custom_lines = [Line2D([0], [0], color='skyblue', lw=4, alpha=0.4),
                Line2D([0], [0], color='red', lw=2, linestyle='--')]
ax.legend(custom_lines, ['Recouvrement (Boules)', 'Nerf (Complexe Simplicial)'])

plt.title(f"Illustration du Théorème du Nerf\nLe graphe rouge (Nerf) capture le 'trou' formé par les boules bleues")
plt.show()