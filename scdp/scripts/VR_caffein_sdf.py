import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
from gudhi import RipsComplex, plot_persistence_diagram


input_path = './Charge_density/data/data_ch.csv'
output_filename = 'clean_data.csv'
output_path = './Charge_density/data/' + output_filename

# Load and clean the data
points = pd.read_csv(input_path, sep=';', skipinitialspace=True)
points.columns = points.columns.str.strip()
points = points[['X', 'Y', 'Z']]         # Keep only the coordinate columns
points = points.astype(float)            # Convert to real numbers
points = points.to_numpy()               # Convert to NumPy array

print(points.shape)


#  Create Vietoris–Rips complex

max_edge_length = 2.0
rips_complex = RipsComplex(points=points, max_edge_length=max_edge_length)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
simplex_tree.compute_persistence()


#  Plot persistence diagram

plot_persistence_diagram(simplex_tree.persistence())
plt.title("Persistence Diagram (Vietoris–Rips Complex, 3D)")
plt.show()


#  Prepare for filtration animation

# Collect simplices and their filtration values
simplices = []
for simplex, filtration in simplex_tree.get_filtration():
    if len(simplex) <= 3:
        simplices.append((filtration, simplex))
simplices.sort()

# Filtration values as animation frames
filtration_values = sorted(set(f for f, _ in simplices))


#  Setup 3D animation figure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_title("3D Vietoris–Rips Filtration with Growing Balls")

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=40, zorder=3)

# For edges and triangles
drawn_edges = []
drawn_triangles = []
drawn_balls = []


#  Animation update function

def update(frame):
    current_filtration = filtration_values[frame]
    ax.set_title(f"Filtration ε = {current_filtration:.2f}")
    
    # Clear previous frame (edges/triangles/balls)
    while drawn_edges:
        edge = drawn_edges.pop()
        edge.remove()
    while drawn_triangles:
        tri = drawn_triangles.pop()
        tri.remove()
    while drawn_balls:
        ball = drawn_balls.pop()
        ball.remove()
    
    # Draw balls (spheres) — here as wireframes
    r = current_filtration / 2
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
    for p in points:
        x = p[0] + r * np.cos(u) * np.sin(v)
        y = p[1] + r * np.sin(u) * np.sin(v)
        z = p[2] + r * np.cos(v)
        ball = ax.plot_surface(x, y, z, color='skyblue', alpha=0.4, linewidth=0, shade=True)  
        drawn_balls.append(ball)

    # Draw new simplices
    for f, simplex in simplices:
        if f == current_filtration:
            if len(simplex) == 2:
                i, j = simplex
                line = ax.plot([points[i][0], points[j][0]],
                               [points[i][1], points[j][1]],
                               [points[i][2], points[j][2]],
                               'b-', alpha=0.7)
                drawn_edges.extend(line)
            elif len(simplex) == 3:
                i, j, k = simplex
                verts = [points[i], points[j], points[k]]
                tri = Poly3DCollection([verts], color='orange', alpha=0.3)
                ax.add_collection3d(tri)
                drawn_triangles.append(tri)


# Run animation

anim = FuncAnimation(fig, update, frames=len(filtration_values), interval=1000, repeat=False)
plt.show()

