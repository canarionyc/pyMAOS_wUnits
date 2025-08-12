# Cantilevered stairway analysis - reads data from JSON file

import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load stairway structure from JSON file
json_file = os.path.join(os.getcwd(),'stairway_structure_Imperial.JSON')
print(f"Loading stairway structure from {json_file}")

with open( json_file, 'r') as file:
    # Remove comments if present in the JSON file
    json_str = ""
    for line in file:
        if '//' not in line:
            json_str += line

    stairway_data = json.loads(json_str)

print(f"Successfully loaded stairway structure: {stairway_data['design_parameters']['name']}")

# Extract structure data
nodes = stairway_data['nodes']
supports = stairway_data['supports']
members = stairway_data['members']
materials = stairway_data['materials']
sections = stairway_data['sections']

print(f"Structure contains {len(nodes)} nodes, {len(members)} members, and {len(supports)} supports")

# Visualize the structure
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create node dictionary for quick lookups
node_dict = {}
for node in nodes:
    node_id = node['id']
    x = float(node['x'].split()[0])
    y = float(node['y'].split()[0])
    z = float(node['z'].split()[0])
    node_dict[node_id] = (x, y, z)

# Plot members
section_colors = {
    1: 'blue',    # Main stringers
    2: 'green',   # Steps/treads
    3: 'cyan',    # Secondary members
    4: 'gray',    # Bracing
    5: 'red'      # Column
}

section_widths = {
    1: 2,         # Main stringers
    2: 1.5,       # Steps/treads
    3: 1,         # Secondary members
    4: 1,         # Bracing
    5: 3          # Column
}

for member in members:
    i_node = member['i_node']
    j_node = member['j_node']
    section = member['section']

    if i_node not in node_dict or j_node not in node_dict:
        print(f"WARNING: Member {member['id']} references non-existent node(s): {i_node}-{j_node}")
        continue

    x1, y1, z1 = node_dict[i_node]
    x2, y2, z2 = node_dict[j_node]

    color = section_colors.get(section, 'black')
    linewidth = section_widths.get(section, 1)

    ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=linewidth)

# Plot nodes
for node_id, (x, y, z) in node_dict.items():
    ax.scatter(x, y, z, color='black', s=20)

# Plot supports with different markers
for support in supports:
    node_id = support['node']
    if node_id not in node_dict:
        print(f"WARNING: Support references non-existent node: {node_id}")
        continue

    x, y, z = node_dict[node_id]

    # Full support (fixed in all directions)
    if support['ux'] == 1 and support['uy'] == 1 and support['uz'] == 1:
        ax.scatter(x, y, z, color='red', s=100, marker='s')
    # Partial support
    else:
        ax.scatter(x, y, z, color='orange', s=80, marker='^')

# Add a wall surface to represent the back wall
# Find the y value for the back of the stairway
back_y = max(pos[1] for pos in node_dict.values())
min_x = min(pos[0] for pos in node_dict.values())
max_x = max(pos[0] for pos in node_dict.values())
min_z = min(pos[2] for pos in node_dict.values())
max_z = max(pos[2] for pos in node_dict.values())

# Add some margin to the wall
margin = 1
wall_x = np.array([min_x-margin, max_x+margin])
wall_z = np.array([min_z-margin, max_z+margin])
wall_x_grid, wall_z_grid = np.meshgrid(wall_x, wall_z)
wall_y_grid = np.ones_like(wall_x_grid) * back_y

# Plot the wall with slight transparency
ax.plot_surface(wall_x_grid, wall_y_grid, wall_z_grid, color='gray', alpha=0.3)

# Create a legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=section_colors[1], lw=section_widths[1], label='Main Stringers'),
    Line2D([0], [0], color=section_colors[2], lw=section_widths[2], label='Steps/Treads'),
    Line2D([0], [0], color=section_colors[4], lw=section_widths[4], label='Bracing'),
    Line2D([0], [0], color=section_colors[5], lw=section_widths[5], label='Column'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Fixed Support'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='Partial Support')
]
ax.legend(handles=legend_elements, loc='upper right')

# Set axis labels and title
ax.set_xlabel('X (ft)')
ax.set_ylabel('Y (ft)')
ax.set_zlabel('Z (ft)')
title = stairway_data['design_parameters']['name']
ax.set_title(title)

# Adjust view angle for better visualization
ax.view_init(elev=20, azim=-60)

# Set equal aspect ratio for a more realistic view
ax.set_box_aspect([max_x-min_x, back_y+1, max_z-min_z])

# Add text annotations for key structural elements
ax.text(0, 0, max_z+0.5, "Upper Landing", color='blue')
ax.text(max_x, 0, min_z-0.5, "Lower Landing", color='blue')
ax.text((max_x-min_x)/2, back_y, max_z/2, "Wall Support", color='black')

# Save figure with better resolution
output_filename = json_file.replace('.JSON', '.png')
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Stairway visualization saved to {output_filename}")

plt.show()

# Print additional structural information
print("\nStructural Information:")
print(f"Top landing position: x=0, z={max_z} ft")
print(f"Bottom landing position: x={max_x} ft, z=0 ft")
print(f"Stairway span: {max_x} ft horizontal, {max_z} ft vertical")
print(f"Number of steps: {len([m for m in members if m['section'] == 2 and 5 <= m['id'] <= 40])}")
