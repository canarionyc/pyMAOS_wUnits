import sys
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import linalg

# Load the bridge definition
bridge_file = 'Pratt_Bridge_3D.YAML'
print(f"Loading 3D bridge from {bridge_file}")
with open(os.path.join(os.path.dirname(__file__), bridge_file), 'r') as file:
    bridge = yaml.safe_load(file)

# Extract design parameters if present
design_params = bridge.get('design_parameters', {})
min_safety_factor = design_params.get('min_safety_factor', 1.0)
print(f"\nDesign minimum safety factor: {min_safety_factor}")

# Extract nodes and members
nodes = bridge['nodes']
members = bridge['members']
supports = bridge['supports']

# Create a dictionary for easier node lookup and a node_id to index mapping
node_dict = {}
node_to_index = {}  # Maps node_id to its position index (0-based)
for i, node in enumerate(nodes):
    node_id = node['id']
    x = float(str(node['x']).split()[0])
    y = float(str(node['y']).split()[0])
    z = float(str(node.get('z', '0 ft')).split()[0])  # Default to 0 if z not present
    node_dict[node_id] = (x, y, z)
    node_to_index[node_id] = i  # Store 0-based index for each node

print(f"Created node index mapping for {len(node_to_index)} nodes")
print(f"First few mappings: {list(node_to_index.items())[:5]}")
print(f"Highest node ID: {max(node_to_index.keys())}")

# Check for duplicate nodes with higher precision
print("\nChecking for duplicate nodes (high precision check)...")
node_positions = {}
duplicate_nodes = []
duplicate_tolerance = 1e-6  # Tolerance for considering nodes as duplicates (in feet)

for node_id, pos in node_dict.items():
    # Round to handle potential floating point precision issues
    pos_key = (round(pos[0]/duplicate_tolerance)*duplicate_tolerance,
               round(pos[1]/duplicate_tolerance)*duplicate_tolerance,
               round(pos[2]/duplicate_tolerance)*duplicate_tolerance)

    if pos_key in node_positions:
        other_node = node_positions[pos_key]
        print(f"WARNING: Nodes {other_node} and {node_id} are at the same position {pos}")
        print(f"  Node {other_node}: ({node_dict[other_node][0]}, {node_dict[other_node][1]}, {node_dict[other_node][2]})")
        print(f"  Node {node_id}: ({pos[0]}, {pos[1]}, {pos[2]})")
        print("This will cause numerical instability!")
        duplicate_nodes.append((other_node, node_id))
    else:
        node_positions[pos_key] = node_id

# Report structural connectivity
print("\nStructural connectivity check:")
print(f"Total nodes: {len(node_dict)}")
print(f"Unique positions: {len(node_positions)}")
if duplicate_nodes:
    print(f"Found {len(duplicate_nodes)} duplicate node pairs:")
    for node1, node2 in duplicate_nodes:
        pos = node_dict[node1]
        print(f"  Nodes {node1} and {node2} at position ({pos[0]}, {pos[1]}, {pos[2]})")
    print("Duplicate nodes should be merged for numerical stability")
else:
    print("No duplicate nodes found - good!")

# Check for disconnected parts
print("\nChecking for member connectivity...")
node_connections = {node_id: set() for node_id in node_dict}
for member in members:
    if member['i_node'] in node_dict and member['j_node'] in node_dict:
        node_connections[member['i_node']].add(member['j_node'])
        node_connections[member['j_node']].add(member['i_node'])

# Find isolated nodes
isolated_nodes = [node_id for node_id, connections in node_connections.items() if len(connections) == 0]
if isolated_nodes:
    print(f"WARNING: Found {len(isolated_nodes)} isolated nodes: {isolated_nodes}")

# Check connectivity between the two spans
# Nodes at x=100 ft should connect the left and right spans
nodes_at_pier = [node_id for node_id, pos in node_dict.items() if abs(pos[0] - 100) < 0.1]
print(f"\nNodes at center pier (x=100 ft): {sorted(nodes_at_pier)}")

# Add function to calculate distances between node pairs
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def print_all_node_distances(node_dict, output_file=None):
    """Calculate and print distances between all node pairs"""
    # Generate all pairs of nodes
    node_pairs = []
    node_ids = sorted(node_dict.keys())

    print("\nCalculating distances between all node pairs...")
    total_pairs = len(node_ids) * (len(node_ids) - 1) // 2
    print(f"Total {total_pairs} unique node pairs")

    for i, node1 in enumerate(node_ids):
        for node2 in node_ids[i+1:]:  # Start from i+1 to avoid duplicates
            point1 = node_dict[node1]
            point2 = node_dict[node2]
            distance = calculate_distance(point1, point2)
            node_pairs.append((node1, node2, distance))

    # Sort by distance (ascending)
    node_pairs.sort(key=lambda x: x[2])

    # Print results (show only first 20 pairs to avoid flooding console)
    print(f"\nNode pair distances (sorted by distance, showing first 20):")
    print(f"{'Node 1':<6} {'Node 2':<6} {'Distance (ft)':<12}")
    print("-" * 30)

    for node1, node2, dist in node_pairs[:20]:
        print(f"{node1:<6} {node2:<6} {dist:<12.3f}")

    # Save all results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"{'Node 1':<6} {'Node 2':<6} {'Distance (ft)':<12}\n")
            f.write("-" * 30 + "\n")
            for node1, node2, dist in node_pairs:
                f.write(f"{node1:<6} {node2:<6} {dist:<12.3f}\n")
        print(f"\nAll node distances saved to {output_file}")

    return node_pairs

# Calculate and save distances between all node pairs
timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
node_distances_file = os.path.join(os.path.dirname(__file__), f"Pratt_Bridge_3D_Node_Distances_{timestamp}.txt")
node_pairs = print_all_node_distances(node_dict, node_distances_file)

# Check for members that should exist but don't
print("\nVerifying critical connections:")
critical_connections = [
    (6, 12, "Central vertical at x=100ft, front"),
    (106, 112, "Central vertical at x=100ft, back"),
    (106, 12, "Diagonal bracing at center pier")
]

for i_node, j_node, description in critical_connections:
    found = any((m['i_node'] == i_node and m['j_node'] == j_node) or
                (m['i_node'] == j_node and m['j_node'] == i_node)
                for m in members)
    if not found:
        print(f"  WARNING: Missing critical member {description} between nodes {i_node} and {j_node}")
    else:
        print(f"  OK: Found {description}")

# Define function to get DOF indices for a node (3D: 3 DOFs per node)
def get_dof(node_id):
    if node_id not in node_to_index:
        raise ValueError(f"Node ID {node_id} not found in node index mapping")
    base = 3 * node_to_index[node_id]  # Use the node index, not the ID
    return base, base + 1, base + 2  # x, y, z DOFs

# Initialize global stiffness matrix and force vector
num_nodes = len(nodes)
dof = 3 * num_nodes  # Degrees of freedom (x, y, z for each node)
K_global = np.zeros((dof, dof))
F_global = np.zeros(dof)

print(f"\n3D Analysis: {num_nodes} nodes, {dof} DOFs")

# Material properties (A36 steel)
E = 29000.0  # ksi - Young's modulus

# Define section properties - now load from YAML if available
if 'sections' in bridge:
    sections = {}
    for section in bridge['sections']:
        sec_id = section['id']
        area_str = str(section['area']).split()[0]  # Extract numeric part
        r_str = str(section['r']).split()[0]        # Extract numeric part
        sections[sec_id] = {
            "area": float(area_str),
            "r": float(r_str)
        }
    print("\nLoaded section properties from YAML:")
    for sec_id, props in sections.items():
        print(f"  Section {sec_id}: Area = {props['area']} in^2, r = {props['r']} in")
else:
    # Default section properties if not in YAML
    sections = {
        1: {"area": 7.65, "r": 4.32},  # W12x26 - area in in^2, r is radius of gyration in inches
        2: {"area": 9.13, "r": 3.47}   # W8x31 - area in in^2, r is radius of gyration in inches
    }
    print("\nUsing default section properties")

# Store member data for later reference
member_data = {}

# Calculate axial stiffness for each member
print("\nAssembling 3D stiffness matrix...")
for member in members:
    member_id = member['id']
    start = member['i_node']
    end = member['j_node']

    # Verify nodes exist in our mapping
    if start not in node_to_index:
        print(f"ERROR: Member {member_id} references non-existent node {start}")
        continue
    if end not in node_to_index:
        print(f"ERROR: Member {member_id} references non-existent node {end}")
        continue

    section_id = member['section']

    # Get section properties
    section_area = sections[section_id]["area"]  # in²

    # Calculate EA in kip-in (E in ksi, A in in² => EA in kip)
    EA = E * section_area

    # Get coordinates
    x1, y1, z1 = node_dict[start]
    x2, y2, z2 = node_dict[end]

    # Calculate length in ft
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length_ft = np.sqrt(dx**2 + dy**2 + dz**2)

    # Check for zero-length members
    if length_ft < 1e-6:
        print(f"ERROR: Member {member_id} has zero length (nodes {start} to {end})")
        print(f"  Node {start}: ({x1}, {y1}, {z1})")
        print(f"  Node {end}: ({x2}, {y2}, {z2})")
        continue  # Skip this member instead of exiting

    length_in = length_ft * 12.0  # Convert to inches

    # Calculate direction cosines for 3D
    cx = dx / length_ft
    cy = dy / length_ft
    cz = dz / length_ft

    # Store member properties
    member_data[member_id] = {
        "start": start,
        "end": end,
        "length_ft": length_ft,
        "length_in": length_in,
        "EA": EA,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "section_id": section_id,
        "section_area": section_area
    }

    # Debug info for first few members
    if member_id <= 3:
        print(f"Member {member_id}: Length = {length_ft:.2f} ft ({length_in:.2f} in), EA = {EA:.2f} kip")
        print(f"  Direction cosines: cx = {cx:.4f}, cy = {cy:.4f}, cz = {cz:.4f}")

    # Local to global transformation matrix for 3D truss element
    # The transformation matrix is 6x6 for 3D (3 DOFs per node)
    T = np.array([
        [cx, cy, cz, 0,  0,  0],
        [0,  0,  0, cx, cy, cz]
    ])

    # Element stiffness matrix in local coordinates (for a 3D truss element)
    # k = EA/L in kip/in
    k_axial = EA / length_in

    k_local = k_axial * np.array([
        [ 1, -1],
        [-1,  1]
    ])

    # Transform to global coordinates: k_global = T^T * k_local * T
    k_global = T.T @ k_local @ T

    # Map local DOFs to global DOFs using node indices, not IDs
    dof_start = get_dof(start)
    dof_end = get_dof(end)
    dof_indices = [*dof_start, *dof_end]

    if member_id <= 3:
        print(f"  DOF indices: {dof_indices}")

    # Assemble into global stiffness matrix
    for i in range(6):
        for j in range(6):
            K_global[dof_indices[i], dof_indices[j]] += k_global[i, j]

# Apply external forces from member loads
print("\nApplying loads...")
if 'member_loads' in bridge:
    for load in bridge['member_loads']:
        member_id = load['member_uid']
        if member_id not in member_data:
            print(f"WARNING: Load references non-existent member {member_id}")
            continue

        member = next(m for m in members if m['id'] == member_id)
        i_node = member['i_node']
        j_node = member['j_node']

        # Convert distributed load to equivalent nodal loads
        wi = float(str(load['wi']).split()[0])  # kips/ft
        wj = float(str(load['wj']).split()[0])  # kips/ft
        length = member_data[member_id]["length_ft"]  # ft

        # For uniform load, apply half to each node in y-direction
        if abs(wi - wj) < 1e-6:  # Uniform load
            force = abs(wi) * length / 2  # kips (make positive)
            _, dof_yi, _ = get_dof(i_node)
            _, dof_yj, _ = get_dof(j_node)

            # Debug info for first few loads
            if member_id <= 3:
                print(f"Load on member {member_id}: w = {wi} kips/ft, L = {length:.2f} ft")
                print(f"  Equivalent nodal forces: {force:.2f} kips (downward) at nodes {i_node} and {j_node}")

            # Apply downward forces (negative in typical structural convention)
            F_global[dof_yi] -= force  # Negative for downward
            F_global[dof_yj] -= force  # Negative for downward
        else:
            # For non-uniform load (simplified approach)
            force_i = abs(wi) * length / 3  # kips
            force_j = abs(wj) * length / 3  # kips
            _, dof_yi, _ = get_dof(i_node)
            _, dof_yj, _ = get_dof(j_node)
            F_global[dof_yi] -= force_i  # Negative for downward
            F_global[dof_yj] -= force_j  # Negative for downward

# Apply support constraints
print("\nApplying support constraints:")
constrained_dofs = []
for support in supports:
    node_id = support['node']

    if node_id not in node_to_index:
        print(f"WARNING: Support references non-existent node {node_id}")
        continue

    dof_x, dof_y, dof_z = get_dof(node_id)

    print(f"Support at node {node_id}: ux={support['ux']}, uy={support['uy']}, uz={support.get('uz', 0)}")

    if support['ux'] == 1:  # Fixed in x-direction
        constrained_dofs.append(dof_x)
        K_global[dof_x, :] = 0
        K_global[:, dof_x] = 0
        K_global[dof_x, dof_x] = 1
        F_global[dof_x] = 0

    if support['uy'] == 1:  # Fixed in y-direction
        constrained_dofs.append(dof_y)
        K_global[dof_y, :] = 0
        K_global[:, dof_y] = 0
        K_global[dof_y, dof_y] = 1
        F_global[dof_y] = 0

    if support.get('uz', 0) == 1:  # Fixed in z-direction
        constrained_dofs.append(dof_z)
        K_global[dof_z, :] = 0
        K_global[:, dof_z] = 0
        K_global[dof_z, dof_z] = 1
        F_global[dof_z] = 0

print(f"Total constrained DOFs: {len(constrained_dofs)}")
print(f"Free DOFs: {dof - len(constrained_dofs)}")

# Check for zero diagonal elements (indicates singularity)
diag_zeros = np.where(np.abs(np.diag(K_global)) < 1e-10)[0]
if len(diag_zeros) > 0:
    print(f"\nWARNING: Zero diagonal elements detected at DOFs: {diag_zeros}")
    for dof_idx in diag_zeros:
        # Find which node this DOF belongs to
        node_idx = dof_idx // 3
        # Find the node_id from the index
        node_id = None
        for nid, idx in node_to_index.items():
            if idx == node_idx:
                node_id = nid
                break
        dof_type = ["x", "y", "z"][dof_idx % 3]
        print(f"  DOF {dof_idx} (Node {node_id}, {dof_type}-direction)")

# Check if the stiffness matrix is symmetric (it should be)
is_symmetric = np.allclose(K_global, K_global.T, rtol=1e-5, atol=1e-8)
print(f"\nStiffness matrix is symmetric: {is_symmetric}")

# Check for rows/columns that are all zeros (except diagonal)
print("\nChecking for zero rows/columns in stiffness matrix...")
zero_rows = []
for i in range(dof):
    row = K_global[i, :]
    if np.sum(np.abs(row)) - np.abs(row[i]) < 1e-10:  # Row has only diagonal element
        zero_rows.append(i)
        # Find which node this DOF belongs to
        node_idx = i // 3
        # Find the node_id from the index
        node_id = None
        for nid, idx in node_to_index.items():
            if idx == node_idx:
                node_id = nid
                break
        dof_type = ["x", "y", "z"][i % 3]
        print(f"  DOF {i} (Node {node_id}, {dof_type}-direction) has no off-diagonal connections")

# Analyze the condition of the matrix
try:
    # Compute the condition number
    cond = np.linalg.cond(K_global)
    print(f"Condition number of stiffness matrix: {cond:.2e}")

    if cond > 1e12:
        print("\nWARNING: Very high condition number detected!")
        print("The structure may have:")
        print("1. Insufficient supports (mechanism)")
        print("2. Disconnected parts")
        print("3. Numerical scaling issues")
        print("4. Duplicate nodes at the same location")

        # Additional diagnostics
        print("\nDiagnostic information:")
        print(f"- Number of duplicate node positions: {len(duplicate_nodes)}")
        print(f"- Number of isolated nodes: {len(isolated_nodes)}")
        print(f"- Number of zero rows/columns: {len(zero_rows)}")

        # Check for specific problematic DOFs
        if len(zero_rows) > 0:
            print("\nNodes with unconnected DOFs:")
            problematic_nodes = set()
            for dof_idx in zero_rows:
                node_idx = dof_idx // 3
                for nid, idx in node_to_index.items():
                    if idx == node_idx:
                        problematic_nodes.add(nid)
                        break
            print(f"  Nodes with issues: {sorted(problematic_nodes)}")

except Exception as e:
    print(f"Could not compute condition number: {e}")
    cond = float('inf')

# Apply scaling to improve conditioning
print("\nAttempting to solve the system...")
try:
    # For very ill-conditioned systems, try regularization
    if cond > 1e14:
        print("System is extremely ill-conditioned. Applying regularization...")
        # Add small diagonal perturbation to improve conditioning
        reg_factor = 1e-8 * np.max(np.abs(np.diag(K_global)))
        K_regularized = K_global + reg_factor * np.eye(dof)

        try:
            displacements = linalg.solve(K_regularized, F_global, assume_a='sym')
            print("Regularized solution successful")
        except:
            # Fall back to least squares
            displacements, residuals, rank, s = linalg.lstsq(K_global, F_global)
            print(f"Least squares solution: rank={rank}/{dof}")
    else:
        # Try direct solution
        displacements = linalg.solve(K_global, F_global, assume_a='sym')
        print("Direct solution successful")

    # Print first few displacement values for debugging
    print("\nNodal displacements:")
    print("Node | dx (in) | dy (in) | dz (in)")
    print("-" * 40)
    for node_id in sorted(list(node_dict.keys()))[:10]:  # First 10 nodes
        dof_x, dof_y, dof_z = get_dof(node_id)
        # Convert to inches for display
        dx_in = displacements[dof_x] * 12
        dy_in = displacements[dof_y] * 12
        dz_in = displacements[dof_z] * 12
        print(f"{node_id:4} | {dx_in:7.4f} | {dy_in:7.4f} | {dz_in:7.4f}")

except Exception as e:
    print(f"All solution methods failed: {e}")
    print("Using zero displacements as fallback")
    displacements = np.zeros(dof)

# Find maximum vertical displacement
max_vert_disp = 0
max_disp_node = None
for node_id in node_dict:
    _, dof_y, _ = get_dof(node_id)
    # Skip nodes that are supported
    is_supported = any(s['node'] == node_id and s['uy'] == 1 for s in supports)
    if not is_supported:
        disp = abs(displacements[dof_y])
        if disp > max_vert_disp:
            max_vert_disp = disp
            max_disp_node = node_id

# Convert to inches for display
max_vert_disp_in = max_vert_disp * 12
print(f"\nMaximum vertical displacement: {max_vert_disp_in:.4f} inches at node {max_disp_node}")

# Calculate member forces
member_forces = {}
member_stresses = {}

# Material properties (A36 steel)
Fy = 36.0    # ksi - Yield strength
Fu = 58.0    # ksi - Ultimate strength
safety_factor_tension = 1.67
safety_factor_compression = 1.67
allowable_tension_stress = Fy / safety_factor_tension      # ksi
allowable_compression_stress_max = Fy / safety_factor_compression  # ksi

# Calculate forces and stresses for each member
for member in members:
    member_id = member['id']
    if member_id not in member_data:
        continue  # Skip members with zero length or missing data

    mem_data = member_data[member_id]
    start = mem_data["start"]
    end = mem_data["end"]
    area = mem_data["section_area"]  # in²
    cx = mem_data["cx"]
    cy = mem_data["cy"]
    cz = mem_data["cz"]

    # Get displacements at both nodes (in feet)
    dof_x1, dof_y1, dof_z1 = get_dof(start)
    dof_x2, dof_y2, dof_z2 = get_dof(end)

    # Calculate relative displacement in the axial direction (in feet)
    delta_u1 = displacements[dof_x1] * cx + displacements[dof_y1] * cy + displacements[dof_z1] * cz
    delta_u2 = displacements[dof_x2] * cx + displacements[dof_y2] * cy + displacements[dof_z2] * cz
    delta_axial = delta_u2 - delta_u1

    # Convert to inches for force calculation
    delta_axial_in = delta_axial * 12

    # Calculate axial force (F = EA/L * delta)
    EA = mem_data["EA"]  # kip
    force = (EA / mem_data["length_in"]) * delta_axial_in  # kips
    member_forces[member_id] = force

    # Calculate axial stress
    stress = force / area  # kips/in² = ksi
    member_stresses[member_id] = stress

    # Debug output for selected members
    if member_id <= 3 or member_id == 32:
        print(f"\nMember {member_id} analysis:")
        print(f"  Length = {mem_data['length_ft']:.2f} ft")
        print(f"  Node {start}: dx = {displacements[dof_x1]*12:.4f} in, dy = {displacements[dof_y1]*12:.4f} in, dz = {displacements[dof_z1]*12:.4f} in")
        print(f"  Node {end}: dx = {displacements[dof_x2]*12:.4f} in, dy = {displacements[dof_y2]*12:.4f} in, dz = {displacements[dof_z2]*12:.4f} in")
        print(f"  Delta axial = {delta_axial_in:.4f} in")
        print(f"  Force = {force:.2f} kips")
        print(f"  Stress = {stress:.2f} ksi")

# Calculate safety factors for all members
member_safety_factors = {}
member_status = {}

# Define zero force threshold
ZERO_FORCE_THRESHOLD = 0.1  # Consider forces below this threshold as "zero"

# Replace the problematic part in the safety factors calculation
for member_id, stress in member_stresses.items():
    section_id = member_data[member_id]["section_id"]
    section = sections[section_id]

    if abs(stress) < ZERO_FORCE_THRESHOLD / section["area"]:
        # Zero/near-zero stress
        safety_factor = float('inf')
        limit = allowable_tension_stress  # Arbitrary for zero-force members
        status = "OK"
    elif stress >= 0:  # Tension
        safety_factor = allowable_tension_stress / abs(stress) if abs(stress) > 0 else float('inf')
        limit = allowable_tension_stress
        status = "OK" if safety_factor >= 1.0 else "OVERSTRESSED"
    else:  # Compression
        # Calculate allowable compression stress based on slenderness
        length_inches = member_data[member_id]["length_in"]
        slenderness = length_inches / section["r"]

        if slenderness > 100:  # For slender members
            critical_stress = (np.pi**2 * E) / (slenderness**2)
            allowable_compression = min(allowable_compression_stress_max, critical_stress/safety_factor_compression)
        else:  # For stocky members
            allowable_compression = allowable_compression_stress_max

        safety_factor = allowable_compression / abs(stress) if abs(stress) > 0 else float('inf')
        limit = allowable_compression
        status = "OK" if safety_factor >= 1.0 else "OVERSTRESSED"

    member_safety_factors[member_id] = safety_factor
    member_status[member_id] = {
        "stress": stress,
        "allowable": limit,
        "safety_factor": safety_factor,
        "status": status
    }

# After calculating safety factors, add a summary of members below minimum
members_below_min_sf = []
zero_force_members = []
for member_id, status_info in member_status.items():
    if status_info["safety_factor"] < min_safety_factor and status_info["safety_factor"] < 100:
        members_below_min_sf.append({
            'id': member_id,
            'sf': status_info["safety_factor"],
            'stress': status_info["stress"],
            'allowable': status_info["allowable"]
        })
    if abs(member_forces[member_id]) < ZERO_FORCE_THRESHOLD:
        zero_force_members.append(member_id)

if members_below_min_sf:
    print(f"\n{len(members_below_min_sf)} members have safety factor below {min_safety_factor}:")
    for mem in sorted(members_below_min_sf, key=lambda x: x['sf'])[:10]:  # Show worst 10
        print(f"  Member {mem['id']}: SF = {mem['sf']:.2f}, Stress = {abs(mem['stress']):.1f} ksi")

if zero_force_members:
    print(f"\n{len(zero_force_members)} zero-force members detected:")
    print(f"  Members: {sorted(zero_force_members)[:20]}")  # Show first 20

# Create figure for the PDF - add timestamp to filenames
# timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')  # Now defined earlier
pdf_filename = os.path.join(os.path.dirname(__file__), f"Pratt_Bridge_3D_Analysis_{timestamp}.pdf")
text_filename = os.path.join(os.path.dirname(__file__), f"Pratt_Bridge_3D_Analysis_{timestamp}.txt")

# Open the text file for writing the report
with open(text_filename, 'w') as txt_report:
    # Write header information
    txt_report.write("===============================================\n")
    txt_report.write("PRATT BRIDGE STRUCTURAL ANALYSIS REPORT\n")
    txt_report.write("===============================================\n")
    txt_report.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    txt_report.write(f"Input File: {bridge_file}\n")
    txt_report.write(f"Design Minimum Safety Factor: {min_safety_factor}\n\n")

    # Write model statistics
    txt_report.write("MODEL STATISTICS\n")
    txt_report.write("------------------------\n")
    txt_report.write(f"Number of nodes: {num_nodes}\n")
    txt_report.write(f"Number of members: {len(members)}\n")
    txt_report.write(f"Number of supports: {len(supports)}\n")
    txt_report.write(f"Degrees of freedom: {dof}\n")
    txt_report.write(f"Constrained DOFs: {len(constrained_dofs)}\n")
    txt_report.write(f"Free DOFs: {dof - len(constrained_dofs)}\n\n")

    # Write material properties
    txt_report.write("MATERIAL PROPERTIES\n")
    txt_report.write("------------------------\n")
    txt_report.write(f"Material: A36 Steel\n")
    txt_report.write(f"Young's modulus: {E} ksi\n")
    txt_report.write(f"Yield strength: {Fy} ksi\n")
    txt_report.write(f"Ultimate strength: {Fu} ksi\n")
    txt_report.write(f"Safety factor (tension): {safety_factor_tension}\n")
    txt_report.write(f"Safety factor (compression): {safety_factor_compression}\n\n")

    # Write section properties
    txt_report.write("SECTION PROPERTIES\n")
    txt_report.write("------------------------\n")
    for sec_id, props in sections.items():
        txt_report.write(f"Section {sec_id}: Area = {props['area']} in², r = {props['r']} in\n")
    txt_report.write("\n")

    # Write analysis results summary
    txt_report.write("ANALYSIS RESULTS SUMMARY\n")
    txt_report.write("------------------------\n")
    txt_report.write(f"Maximum vertical displacement: {max_vert_disp_in:.4f} inches at node {max_disp_node}\n")
    txt_report.write(f"Stiffness matrix condition number: {cond:.2e}\n")

    # Calculate critical values
    critical_sf = float('inf')
    critical_member = None
    overstressed_count = 0
    for member_id, status_info in member_status.items():
        sf = status_info["safety_factor"]
        if sf < critical_sf and sf < 100:
            critical_sf = sf
            critical_member = member_id
        if status_info["status"] == "OVERSTRESSED":
            overstressed_count += 1

    txt_report.write(f"Number of tension members: {sum(1 for f in member_forces.values() if f > ZERO_FORCE_THRESHOLD)}\n")
    txt_report.write(f"Number of compression members: {sum(1 for f in member_forces.values() if f < -ZERO_FORCE_THRESHOLD)}\n")
    txt_report.write(f"Number of zero-force members: {sum(1 for f in member_forces.values() if abs(f) <= ZERO_FORCE_THRESHOLD)}\n")
    txt_report.write(f"Overstressed members: {overstressed_count}\n")
    txt_report.write(f"Maximum member force: {max(abs(f) for f in member_forces.values()):.2f} kips\n")
    txt_report.write(f"Maximum member stress: {max(abs(s) for s in member_stresses.values()):.2f} ksi\n")
    txt_report.write(f"Minimum safety factor: {critical_sf:.2f} (Member #{critical_member})\n\n")

    if cond > 1e12:
        txt_report.write("\nNUMERICAL STABILITY WARNING:\n")
        txt_report.write("--------------------------------\n")
        txt_report.write(f"Condition number ({cond:.2e}) indicates severe numerical instability.\n")
        txt_report.write("\nPossible causes:\n")
        if duplicate_nodes:
            txt_report.write(f"1. Found {len(duplicate_nodes)} duplicate node pairs:\n")
            for node1, node2 in duplicate_nodes[:5]:  # Show first 5
                txt_report.write(f"   - Nodes {node1} and {node2} at position {node_dict[node1]}\n")
        if isolated_nodes:
            txt_report.write(f"2. Found {len(isolated_nodes)} isolated nodes\n")
        if zero_rows:
            txt_report.write(f"3. Found {len(zero_rows)} DOFs with no off-diagonal connections\n")
        txt_report.write("\nRecommended fixes:\n")
        txt_report.write("- Merge duplicate nodes into single nodes\n")
        txt_report.write("- Ensure proper connectivity between structural parts\n")
        txt_report.write("- Verify all supports are correctly defined\n")
        txt_report.write("- Check member connectivity references\n\n")

    # Write detailed member results
    txt_report.write("DETAILED MEMBER RESULTS\n")
    txt_report.write("------------------------\n")
    txt_report.write("Member | Nodes (i-j) | Force (kips) | Stress (ksi) | Type | S.F. | Status\n")
    txt_report.write("-" * 80 + "\n")

    for member in sorted(members, key=lambda m: m['id']):
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        force = member_forces[member_id]
        stress = member_stresses[member_id]
        sf = member_safety_factors[member_id]
        status = member_status[member_id]["status"]

        # Fix the text report formatting to use only ASCII
        if abs(force) < ZERO_FORCE_THRESHOLD:
            force_type = "ZERO"
            sf_text = "inf"
        elif force > 0:
            force_type = "TENSION"
            sf_text = f"{sf:.2f}"  # Removed the sf<100 condition
        else:
            force_type = "COMPRESSION"
            sf_text = f"{sf:.2f}"  # Removed the sf<100 condition

        txt_report.write(
            f"{member_id:6} | {i_node:2}-{j_node:<2} | {abs(force):11.2f} | {abs(stress):10.2f} | {force_type:10} | {sf_text:>5} | {status}\n")

    txt_report.write("\n\n")

    # Write nodal displacements
    txt_report.write("NODAL DISPLACEMENTS\n")
    txt_report.write("------------------------\n")
    txt_report.write("Node | dx (inches) | dy (inches) | dz (inches)\n")
    txt_report.write("-" * 40 + "\n")

    for node_id in sorted(node_dict.keys()):
        dof_x, dof_y, dof_z = get_dof(node_id)
        dx_in = displacements[dof_x] * 12
        dy_in = displacements[dof_y] * 12
        dz_in = displacements[dof_z] * 12
        txt_report.write(f"{node_id:4} | {dx_in:11.4f} | {dy_in:11.4f} | {dz_in:11.4f}\n")

    # Add a section for closest node pairs
    txt_report.write("\n\nCLOSEST NODE PAIRS\n")
    txt_report.write("------------------------\n")
    txt_report.write("Node 1 | Node 2 | Distance (ft)\n")
    txt_report.write("-" * 40 + "\n")

    # Print the 30 closest pairs (excluding zero distances which might be duplicates)
    closest_pairs = [pair for pair in node_pairs if pair[2] > 1e-6]
    for node1, node2, dist in closest_pairs[:30]:
        txt_report.write(f"{node1:6} | {node2:6} | {dist:10.3f}\n")

    txt_report.write(f"\nNote: Complete list of node distances available in {os.path.basename(node_distances_file)}\n\n")

    txt_report.write("\n\n")
    txt_report.write("RECOMMENDATIONS FOR STRUCTURE REFINEMENT\n")
    txt_report.write("----------------------------------------\n")

    if members_below_min_sf:
        txt_report.write(f"1. {len(members_below_min_sf)} members have safety factor below the design minimum of {min_safety_factor}:\n")
        for mem in sorted(members_below_min_sf, key=lambda x: x['sf'])[:20]:  # Show worst 20
            member = next(m for m in members if m['id'] == mem['id'])
            txt_report.write(f"   - Member {mem['id']} (nodes {member['i_node']}-{member['j_node']}): SF = {mem['sf']:.2f}\n")

        # Calculate required section increase
        worst_sf = min(mem['sf'] for mem in members_below_min_sf)
        required_increase = min_safety_factor / worst_sf
        txt_report.write(f"\n   Worst member requires approximately {required_increase:.1f}x increase in capacity.\n")
        txt_report.write(f"   Consider using larger sections or reducing the loading.\n")
    else:
        txt_report.write("1. All members meet the minimum safety factor requirement.\n")
        txt_report.write("   Consider optimizing the structure by reducing sections for members with high safety factors.\n")

    if max_vert_disp_in > 2.0:  # Assuming 2 inches as a reasonable deflection limit
        txt_report.write(f"2. Consider stiffening the structure to reduce the maximum deflection (currently {max_vert_disp_in:.4f} inches).\n")

    if cond > 1e12:
        txt_report.write("3. Fix numerical stability issues by reviewing structure connectivity and supports.\n")

    txt_report.write("\nReport generated by pyMAOS Truss Analysis Tool\n")

print(f"Text report saved to: {text_filename}")

with PdfPages(pdf_filename) as pdf:
    # 1. BRIDGE ANALYSIS SUMMARY page
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')

    # Add overall bridge status text - avoid using unicode characters
    if overstressed_count == 0:
        status_text = "BRIDGE IS STRUCTURALLY ADEQUATE"
        status_color = 'green'
    else:
        status_text = f"WARNING: BRIDGE HAS {overstressed_count} OVERSTRESSED MEMBERS"
        status_color = 'red'

    plt.figtext(0.5, 0.9, "BRIDGE ANALYSIS SUMMARY", ha='center', fontsize=16, weight='bold')
    plt.figtext(0.5, 0.85, status_text, ha='center', fontsize=14, weight='bold', color=status_color)

    # Add displacement information
    plt.figtext(0.5, 0.78, f"Maximum Vertical Displacement: {max_vert_disp_in:.4f} inches at Node {max_disp_node}",
                ha='center', fontsize=12)

    # Add summary statistics
    critical_text = f"Most critical member: #{critical_member} (Safety Factor: {critical_sf:.2f})"
    plt.figtext(0.5, 0.73, critical_text, ha='center', fontsize=12)

    # Add a summary table of key statistics
    summary_data = [
        ["Total Members", f"{len(members)}"],
        ["Tension Members", f"{sum(1 for f in member_forces.values() if f > ZERO_FORCE_THRESHOLD)}"],
        ["Compression Members", f"{sum(1 for f in member_forces.values() if f < -ZERO_FORCE_THRESHOLD)}"],
        ["Zero-Force Members", f"{sum(1 for f in member_forces.values() if abs(f) <= ZERO_FORCE_THRESHOLD)}"],
        ["Overstressed Members", f"{overstressed_count}"],
        ["Maximum Member Force", f"{max(abs(f) for f in member_forces.values()):.2f} kips"],
        ["Maximum Member Stress", f"{max(abs(s) for s in member_stresses.values()):.2f} ksi"],
        ["Minimum Safety Factor", f"{critical_sf:.2f}"],
        ["Maximum Displacement", f"{max_vert_disp_in:.4f} inches"],
    ]

    # Create a table in the middle of the page
    ax = plt.subplot(111)
    ax.axis('off')
    tbl = plt.table(cellText=summary_data, colLabels=["Metric", "Value"],
                    loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Add design criteria info
    criteria = (
        f"Design Criteria:\n"
        f"- Material: A36 Steel (Fy = 36 ksi)\n"
        f"- Allowable tension stress: {allowable_tension_stress:.1f} ksi\n"
        f"- Allowable compression stress: Based on member slenderness\n"
        f"- Safety factor: {safety_factor_tension:.2f}"
    )
    plt.figtext(0.5, 0.2, criteria, ha='center', fontsize=10,
                bbox=dict(facecolor='#eeeeee', alpha=0.7, boxstyle='round,pad=0.5'))

    # Add timestamp
    plt.figtext(0.5, 0.1, f"Analysis completed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=8)

    # Save the summary page to PDF
    pdf.savefig(fig)
    plt.close(fig)

    # 2. 3D Structure Visualization
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot members with color based on safety factors
    cmap = plt.colormaps['RdYlGn']  # Red-Yellow-Green colormap
    norm = plt.Normalize(0.5, 2.0)

    for member in members:
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        safety_factor = member_safety_factors[member_id]
        safety_factor_capped = min(safety_factor, 5.0)  # Cap extremely high values

        x1, y1, z1 = node_dict[i_node]
        x2, y2, z2 = node_dict[j_node]

        color = cmap(norm(safety_factor_capped))

        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=2)

    # Plot nodes
    for node_id, (x, y, z) in node_dict.items():
        ax.scatter(x, y, z, c='black', s=20)

    # Plot supports with different markers
    for support in supports:
        node_id = support['node']
        x, y, z = node_dict[node_id]
        if support['ux'] == 1 and support['uy'] == 1 and support.get('uz', 0) == 1:
            ax.scatter(x, y-5, z, c='red', s=100, marker='s')  # Fixed support
        else:
            ax.scatter(x, y-5, z, c='blue', s=100, marker='o')  # Roller support

    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel('Z (ft)')
    ax.set_title('3D Pratt Truss Bridge - Safety Factor Visualization')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Safety Factor', shrink=0.6)

    # Set aspect ratio
    ax.set_box_aspect([2, 0.5, 0.5])  # Adjust based on bridge proportions

    pdf.savefig(fig)
    plt.close(fig)

    # 3. Member Forces - 3D visualization
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot members with color based on tension/compression
    for member in members:
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        force = member_forces[member_id]

        x1, y1, z1 = node_dict[i_node]
        x2, y2, z2 = node_dict[j_node]

        # Color based on force type
        if abs(force) < ZERO_FORCE_THRESHOLD:
            color = 'black'
            linewidth = 1
        elif force > 0:
            color = 'red'  # Tension
            linewidth = 2 + min(3, abs(force)/50)
        else:
            color = 'blue'  # Compression
            linewidth = 2 + min(3, abs(force)/50)

        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=linewidth)

    # Add legend
    from matplotlib.lines import Line2D
    red_line = Line2D([0], [0], color='red', linewidth=3, label='Tension')
    blue_line = Line2D([0], [0], color='blue', linewidth=3, label='Compression')
    black_line = Line2D([0], [0], color='black', linewidth=1, label='Zero Force')
    ax.legend(handles=[red_line, blue_line, black_line], loc='upper right')

    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel('Z (ft)')
    ax.set_title('3D Pratt Truss Bridge - Member Forces')
    ax.set_box_aspect([2, 0.5, 0.5])

    pdf.savefig(fig)
    plt.close(fig)

    # 4. 3D Displacement visualization
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate max displacement for scaling
    max_disp_magnitude = 0
    for node_id in node_dict:
        dof_x, dof_y, dof_z = get_dof(node_id)
        disp_magnitude = np.sqrt(displacements[dof_x]**2 + displacements[dof_y]**2 + displacements[dof_z]**2)
        max_disp_magnitude = max(max_disp_magnitude, disp_magnitude)

    # Calculate scale factor
    max_span = max(node_dict[n][0] for n in node_dict) - min(node_dict[n][0] for n in node_dict)
    target_max_disp = max_span * 0.05

    if max_disp_magnitude > 1e-10:
        scale_factor = target_max_disp / max_disp_magnitude
        scale_magnitude = 10 ** int(np.log10(scale_factor))
        scale_factor = round(scale_factor / scale_magnitude) * scale_magnitude
    else:
        scale_factor = 50.0

    # Draw original structure
    for member in members:
        i_node = member['i_node']
        j_node = member['j_node']
        x1, y1, z1 = node_dict[i_node]
        x2, y2, z2 = node_dict[j_node]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='lightgray', linewidth=1, alpha=0.5)

    # Draw displaced structure
    for member in members:
        i_node = member['i_node']
        j_node = member['j_node']
        x1, y1, z1 = node_dict[i_node]
        x2, y2, z2 = node_dict[j_node]

        # Get displacements
        dof_x1, dof_y1, dof_z1 = get_dof(i_node)
        dof_x2, dof_y2, dof_z2 = get_dof(j_node)

        # Apply scaled displacements
        x1_new = x1 + displacements[dof_x1] * scale_factor
        y1_new = y1 + displacements[dof_y1] * scale_factor
        z1_new = z1 + displacements[dof_z1] * scale_factor
        x2_new = x2 + displacements[dof_x2] * scale_factor
        y2_new = y2 + displacements[dof_y2] * scale_factor
        z2_new = z2 + displacements[dof_z2] * scale_factor

        ax.plot([x1_new, x2_new], [y1_new, y2_new], [z1_new, z2_new], color='blue', linewidth=1.5)

    # Highlight max displacement node
    if max_disp_node:
        x, y, z = node_dict[max_disp_node]
        dof_x, dof_y, dof_z = get_dof(max_disp_node)
        x_new = x + displacements[dof_x] * scale_factor
        y_new = y + displacements[dof_y] * scale_factor
        z_new = z + displacements[dof_z] * scale_factor
        ax.scatter(x_new, y_new, z_new, c='red', s=100)

    # Format scale factor
    if scale_factor >= 1000:
        scale_factor_text = f"{scale_factor/1000:.1f}k"
    elif scale_factor >= 100:
        scale_factor_text = f"{int(scale_factor)}"
    else:
        scale_factor_text = f"{scale_factor:.1f}"

    ax.set_xlabel('X (ft)')
    ax.set_ylabel('Y (ft)')
    ax.set_zlabel('Z (ft)')
    ax.set_title(f'3D Pratt Truss Bridge - Displacement (Scale Factor: {scale_factor_text}x)')
    ax.set_box_aspect([2, 0.5, 0.5])

    # Add legend
    gray_line = Line2D([0], [0], color='lightgray', linewidth=1, alpha=0.5, label='Original')
    blue_line = Line2D([0], [0], color='blue', linewidth=1.5, label='Displaced')
    ax.legend(handles=[gray_line, blue_line], loc='upper right')

    pdf.savefig(fig)
    plt.close(fig)

    # Add PDF metadata
    d = pdf.infodict()
    d['Title'] = 'Pratt Bridge Structural Analysis'
    d['Author'] = 'pyMAOS Truss Analysis Tool'
    d['Subject'] = 'Structural analysis and reliability assessment'
    d['Keywords'] = 'truss, structural analysis, bridge, reliability'
    d['CreationDate'] = datetime.datetime.now()
    d['ModDate'] = datetime.datetime.now()

# Print summary of results to console
print("\nMember Forces Summary:")
print("----------------------")
print("Member | Force (kips) | Type")
print("----------------------")
for member in sorted(members, key=lambda m: m['id']):
    member_id = member['id']
    force = member_forces[member_id]
    if abs(force) < ZERO_FORCE_THRESHOLD:
        force_type = "ZERO"
    elif force > 0:
        force_type = "TENSION"
    else:
        force_type = "COMPRESSION"
    print(f"{member_id:6} | {abs(force):11.2f} | {force_type}")

# Print numerical stability information
print("\nNumerical Stability Info:")
print("-----------------------")
print(f"Matrix size: {dof}×{dof}")
print(f"Condition number: {cond:.2e}")
print(f"Number of members: {len(member_data)}")
print(f"Maximum force magnitude: {max(abs(f) for f in member_forces.values()):.2f} kips")
print(f"Maximum displacement: {max_vert_disp_in:.4f} inches")

if cond > 1e12:
    print("\nWARNING: High condition number indicates numerical instability.")
    print("\nRECOMMENDATIONS:")
    if duplicate_nodes:
        for node1, node2 in duplicate_nodes[:3]:  # Show first 3 duplicates
            pos = node_dict[node1]
            print(f"1. The duplicate nodes {node1} and {node2} at ({pos[0]}, {pos[1]}, {pos[2]}) should be merged")
        print("2. Update member connectivity to reference the merged nodes")
    else:
        print("1. Check for nearly-coincident nodes that might be causing instability")
        print("2. Verify member connectivity for proper structural integrity")
    print("3. Ensure all supports are properly defined")
    print("4. Check for any disconnected parts of the structure")

# Display the plot on screen
plt.show()
