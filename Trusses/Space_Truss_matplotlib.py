import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import glob

def visualize_truss(yaml_file):
    # Extract filename without extension for saving PNG
    base_filename = os.path.splitext(yaml_file)[0]
    png_filename = base_filename + '.png'

    print(f"Processing {yaml_file}...")

    # Load the bridge definition
    with open(yaml_file, 'r') as file:
        bridge = yaml.safe_load(file)

    # Extract nodes and members
    nodes = bridge['nodes']
    members = bridge['members']
    supports = bridge.get('supports', [])  # Use empty list if no supports defined

    # Create a dictionary for easier node lookup
    node_dict = {}
    for node in nodes:
        node_id = node['id']
        x = float(str(node['x']).split()[0])
        y = float(str(node['y']).split()[0])
        z = float(str(node['z']).split()[0])
        node_dict[node_id] = (x, z, y)  # Swap Y and Z to make Z point upwards

    # Create 3D figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot members with different colors based on type
    for member in members:
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        section = member.get('section', 1)  # Default to 1 if not specified

        # Get node coordinates (x, z, y) order due to axis swap
        x1, z1, y1 = node_dict[i_node]
        x2, z2, y2 = node_dict[j_node]

        # Determine color and line width based on member type
        if section == 1:  # Chord members
            color = 'blue'
            linewidth = 2
        elif section == 2:  # Verticals and diagonals
            color = 'green'
            linewidth = 1.5
        elif section == 3:  # Transverse beams
            color = 'red'
            linewidth = 2
        else:  # Transverse bracing
            color = 'orange'
            linewidth = 1

        # Plot the member
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=linewidth)

    # Plot nodes
    for node_id, (x, z, y) in node_dict.items():
        ax.scatter(x, y, z, color='black', s=20)

    # Plot supports with different marker
    for support in supports:
        node_id = support['node']
        x, z, y = node_dict[node_id]
        ax.scatter(x, y, z, color='red', s=50, marker='s')

    # Set plot limits and labels
    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel('Z (ft)')  # Z is now up
    ax.set_title(f'3D Truss Structure: {os.path.basename(yaml_file)}')

    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=-45)

    # Set equal aspect ratio
    max_range = np.array([
        np.max([x for x, _, _ in node_dict.values()]) - np.min([x for x, _, _ in node_dict.values()]),
        np.max([y for _, _, y in node_dict.values()]) - np.min([y for _, _, y in node_dict.values()]),
        np.max([z for _, z, _ in node_dict.values()]) - np.min([z for _, z, _ in node_dict.values()])
    ]).max() / 2.0

    mid_x = (np.max([x for x, _, _ in node_dict.values()]) + np.min([x for x, _, _ in node_dict.values()])) * 0.5
    mid_y = (np.max([y for _, _, y in node_dict.values()]) + np.min([y for _, _, y in node_dict.values()])) * 0.5
    mid_z = (np.max([z for _, z, _ in node_dict.values()]) + np.min([z for _, z, _ in node_dict.values()])) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add a legend
    import matplotlib.lines as mlines
    chord = mlines.Line2D([], [], color='blue', linewidth=2, label='Main Chords')
    vertical = mlines.Line2D([], [], color='green', linewidth=1.5, label='Verticals & Diagonals')
    transverse = mlines.Line2D([], [], color='red', linewidth=2, label='Transverse Beams')
    bracing = mlines.Line2D([], [], color='orange', linewidth=1, label='Cross Bracing')
    ax.legend(handles=[chord, vertical, transverse, bracing], loc='upper right')

    plt.tight_layout()

    # Save to PNG file
    print(f"Saving visualization to {png_filename}")
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

if __name__ == '__main__':
    # Find all YAML files matching the pattern
    yaml_files = glob.glob("*_Space_Truss.YAML")
    if not yaml_files:
        print("No files matching '*_Space_Truss.YAML' found in the current directory.")
    else:
        print(f"Found {len(yaml_files)} truss files to process.")
        for yaml_file in yaml_files:
            visualize_truss(yaml_file)
        print("Processing complete!")