import numpy as np

# Define nodes and their coordinates
nodes = {
    1: (0, 0), 2: (20, 0), 3: (40, 0), 4: (60, 0), 5: (80, 0), 6: (100, 0),
    7: (0, 20), 8: (20, 20), 9: (40, 20), 10: (60, 20), 11: (80, 20), 12: (100, 20),
    13: (100, 0), 14: (120, 0), 15: (140, 0), 16: (160, 0), 17: (180, 0), 18: (200, 0),
    19: (100, 20), 20: (120, 20), 21: (140, 20), 22: (160, 20), 23: (180, 20), 24: (200, 20)
}

# Define members as (start_node, end_node)
members = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # Bottom chord (left span)
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),  # Top chord (left span)
    (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12),  # Verticals (left span)
    (7, 2), (8, 3), (9, 4), (10, 5), (11, 6),  # Diagonals (left span)
    (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),  # Bottom chord (right span)
    (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),  # Top chord (right span)
    (13, 19), (14, 20), (15, 21), (16, 22), (17, 23), (18, 24),  # Verticals (right span)
    (20, 13), (21, 14), (22, 15), (23, 16), (24, 17)  # Diagonals (right span)
]

# Define external forces (node_id: (Fx, Fy))
external_forces = {
    1: (0, 0), 6: (0, 0), 18: (0, 0),  # Supports
    2: (0, -2), 3: (0, -2), 4: (0, -2), 5: (0, -2),  # Deck loads (left span)
    14: (0, -2), 15: (0, -2), 16: (0, -2), 17: (0, -2)  # Deck loads (right span)
}

# Define support reactions (node_id: (Rx, Ry))
support_reactions = {
    1: (1, 1), 6: (1, 1), 18: (0, 1)  # Pinned and roller supports
}

# Initialize global stiffness matrix and force vector
num_nodes = len(nodes)
dof = 2 * num_nodes  # Degrees of freedom (x and y for each node)
K_global = np.zeros((dof, dof))
F_global = np.zeros(dof)

# Helper function to get DOF indices for a node
def get_dof(node_id):
    return 2 * (node_id - 1), 2 * (node_id - 1) + 1

# Assemble global stiffness matrix and force vector
for member in members:
    start, end = member
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    cos = (x2 - x1) / length
    sin = (y2 - y1) / length

    # Local stiffness matrix for the member
    k_local = (1 / length) * np.array([
        [cos**2, cos*sin, -cos**2, -cos*sin],
        [cos*sin, sin**2, -cos*sin, -sin**2],
        [-cos**2, -cos*sin, cos**2, cos*sin],
        [-cos*sin, -sin**2, cos*sin, sin**2]
    ])

    # Map local stiffness to global stiffness
    dof_start = get_dof(start)
    dof_end = get_dof(end)
    dof_indices = [*dof_start, *dof_end]
    for i in range(4):
        for j in range(4):
            K_global[dof_indices[i], dof_indices[j]] += k_local[i, j]

# Apply external forces
for node_id, (fx, fy) in external_forces.items():
    dof_x, dof_y = get_dof(node_id)
    F_global[dof_x] += fx
    F_global[dof_y] += fy

# Apply support constraints
for node_id, (rx, ry) in support_reactions.items():
    dof_x, dof_y = get_dof(node_id)
    if rx:
        K_global[dof_x, :] = 0
        K_global[:, dof_x] = 0
        K_global[dof_x, dof_x] = 1
        F_global[dof_x] = 0
    if ry:
        K_global[dof_y, :] = 0
        K_global[:, dof_y] = 0
        K_global[dof_y, dof_y] = 1
        F_global[dof_y] = 0

# Solve for displacements
displacements = np.linalg.solve(K_global, F_global)

# Calculate member forces
member_forces = []
for member in members:
    start, end = member
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    cos = (x2 - x1) / length
    sin = (y2 - y1) / length

    dof_start = get_dof(start)
    dof_end = get_dof(end)
    dof_indices = [*dof_start, *dof_end]

    # Extract displacements for the member
    u = displacements[dof_indices]

    # Calculate axial force in the member
    force = (1 / length) * np.dot(
        np.array([-cos, -sin, cos, sin]),
        u
    )
    member_forces.append((member, force))

# Print results
print("Node Displacements:")
for i, (x, y) in enumerate(nodes.values(), start=1):
    print(f"Node {i}: dx = {displacements[2*(i-1)]:.4f}, dy = {displacements[2*(i-1)+1]:.4f}")

print("\nMember Forces:")
for member, force in member_forces:
    print(f"Member {member}: Force = {force:.4f} kip")