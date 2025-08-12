import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm

# Load the bridge definition
bridge_file = 'Pratt_Bridge.YAML'
print(f"Loading bridge from {bridge_file}")
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

# Create a dictionary for easier node lookup
node_dict = {}
for node in nodes:
    node_id = node['id']
    x = float(str(node['x']).split()[0])
    y = float(str(node['y']).split()[0])
    z = float(str(node.get('z', '0 ft')).split()[0])  # Default to 0 if z not present
    node_dict[node_id] = (x, y, z)

# Check for duplicate nodes
print("\nChecking for duplicate nodes...")
node_positions = {}
for node_id, pos in node_dict.items():
    pos_key = (pos[0], pos[1])  # Use only x,y for 2D
    if pos_key in node_positions:
        raise ValueError(f"WARNING: Nodes {node_positions[pos_key]} and {node_id} are at the same position {pos_key}"+"This will cause numerical instability!")
    else:
        node_positions[pos_key] = node_id

# Define function to get DOF indices for a node
def get_dof(node_id):
    return 2 * (node_id - 1), 2 * (node_id - 1) + 1

# Initialize global stiffness matrix and force vector
num_nodes = len(nodes)
dof = 2 * num_nodes  # Degrees of freedom (x and y for each node)
K_global = np.zeros((dof, dof))
F_global = np.zeros(dof)

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
print("\nAssembling stiffness matrix...")
for member in members:
    member_id = member['id']
    start = member['i_node']
    end = member['j_node']
    section_id = member['section']

    # Get section properties
    section_area = sections[section_id]["area"]  # in²

    # Calculate EA in kip-in (E in ksi, A in in² => EA in kip)
    EA = E * section_area

    # Get coordinates
    x1, y1, _ = node_dict[start]
    x2, y2, _ = node_dict[end]

    # Calculate length in ft
    dx = x2 - x1
    dy = y2 - y1
    length_ft = np.sqrt(dx**2 + dy**2)

    # Check for zero-length members
    if length_ft < 1e-6:
        print(f"ERROR: Member {member_id} has zero length (nodes {start} to {end})")
        print(f"  Node {start}: ({x1}, {y1})")
        print(f"  Node {end}: ({x2}, {y2})")
        sys.exit(1)

    length_in = length_ft * 12.0  # Convert to inches

    # Calculate direction cosines
    cos = dx / length_ft
    sin = dy / length_ft

    # Store member properties
    member_data[member_id] = {
        "start": start,
        "end": end,
        "length_ft": length_ft,
        "length_in": length_in,
        "EA": EA,
        "cos": cos,
        "sin": sin,
        "section_id": section_id,
        "section_area": section_area
    }

    # Debug info for first few members
    if member_id <= 3:
        print(f"Member {member_id}: Length = {length_ft:.2f} ft ({length_in:.2f} in), EA = {EA:.2f} kip")
        print(f"  Direction: cos = {cos:.4f}, sin = {sin:.4f}")

    # Local to global transformation matrix
    T = np.array([
        [ cos, sin, 0,   0],
        [-sin, cos, 0,   0],
        [0,   0,   cos, sin],
        [0,   0,  -sin, cos]
    ])

    # Element stiffness matrix in local coordinates (for a truss element)
    # k = EA/L in kip/in
    k_axial = EA / length_in

    k_local = np.array([
        [ k_axial, 0, -k_axial, 0],
        [0, 0,  0, 0],
        [-k_axial, 0, k_axial, 0],
        [0, 0,  0, 0]
    ])

    # Transform to global coordinates
    k_global = T.T @ k_local @ T
    print(f"member id: {member_id}", k_global, sep="\n")
    # Map local DOFs to global DOFs
    dof_start = get_dof(start)
    dof_end = get_dof(end)
    dof_indices = [*dof_start, *dof_end]; print(f"dof_indices: {dof_indices}")

    # Assemble into global stiffness matrix
    for i in range(4):
        for j in range(4):
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
            _, dof_yi = get_dof(i_node)
            _, dof_yj = get_dof(j_node)

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
            _, dof_yi = get_dof(i_node)
            _, dof_yj = get_dof(j_node)
            F_global[dof_yi] -= force_i  # Negative for downward
            F_global[dof_yj] -= force_j  # Negative for downward

# Apply support constraints
print("\nApplying support constraints:")
constrained_dofs = []
for support in supports:
    node_id = support['node']
    dof_x, dof_y = get_dof(node_id)

    print(f"Support at node {node_id}: ux={support['ux']}, uy={support['uy']}")

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

print(f"Total constrained DOFs: {len(constrained_dofs)}")
print(f"Free DOFs: {dof - len(constrained_dofs)}")

# Check for zero diagonal elements (indicates singularity)
diag_zeros = np.where(np.abs(np.diag(K_global)) < 1e-10)[0]
if len(diag_zeros) > 0:
    print(f"\nWARNING: Zero diagonal elements detected at DOFs: {diag_zeros}")
    for dof_idx in diag_zeros:
        node_id = dof_idx // 2 + 1
        dof_type = "x" if dof_idx % 2 == 0 else "y"
        print(f"  Node {node_id}, direction {dof_type}")

        # Check if this DOF is connected to any members
        connected_members = []
        for mem_id, mem_data in member_data.items():
            if mem_data["start"] == node_id or mem_data["end"] == node_id:
                connected_members.append(mem_id)
        print(f"    Connected to members: {connected_members}")

# Check if the stiffness matrix is symmetric (it should be)
is_symmetric = np.allclose(K_global, K_global.T, rtol=1e-5, atol=1e-8)
print(f"\nStiffness matrix is symmetric: {is_symmetric}")

# Analyze the condition of the matrix
try:
    # Compute the condition number
    cond = np.linalg.cond(K_global)
    print(f"Condition number of stiffness matrix: {cond:.2e}")

    if cond > 1e12:
        print("\nWARNING: Very high condition number detected!")
        print("The structure may have:")
        print("1. Duplicate nodes at the same location")
        print("2. Insufficient supports (mechanism)")
        print("3. Disconnected parts")
        print("4. Numerical scaling issues")

        # Find the ratio between max and min non-zero diagonal elements
        diag = np.abs(np.diag(K_global))
        nonzero_diag = diag[diag > 1e-10]
        if len(nonzero_diag) > 0:
            diag_ratio = np.max(nonzero_diag) / np.min(nonzero_diag)
            print(f"\nDiagonal scaling ratio: {diag_ratio:.2e}")

except Exception as e:
    print(f"Could not compute condition number: {e}")
    cond = float('inf')

# Apply scaling to improve conditioning
print("\nAttempting to solve the system...")
try:
    # Use a more robust solver from scipy
    from scipy import linalg

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

    # Convert displacements from inches to feet for consistency
    # Actually, the displacements are already in feet since we used consistent units

    # Print first few displacement values for debugging
    print("\nNodal displacements:")
    print("Node | dx (in) | dy (in)")
    print("-" * 30)
    for i in range(1, min(11, num_nodes + 1)):
        dof_x, dof_y = get_dof(i)
        # Convert to inches for display
        dx_in = displacements[dof_x] * 12
        dy_in = displacements[dof_y] * 12
        print(f"{i:4} | {dx_in:7.4f} | {dy_in:7.4f}")

except Exception as e:
    print(f"All solution methods failed: {e}")
    print("Using zero displacements as fallback")
    displacements = np.zeros(dof)

# Find maximum vertical displacement
max_vert_disp = 0
max_disp_node = None
for node_id in node_dict:
    _, dof_y = get_dof(node_id)
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
        continue  # Skip members with zero length

    mem_data = member_data[member_id]
    start = mem_data["start"]
    end = mem_data["end"]
    area = mem_data["section_area"]  # in²
    cos = mem_data["cos"]
    sin = mem_data["sin"]

    # Get displacements at both nodes (in feet)
    dof_x1, dof_y1 = get_dof(start)
    dof_x2, dof_y2 = get_dof(end)

    # Calculate relative displacement in the axial direction (in feet)
    delta_u1 = displacements[dof_x1] * cos + displacements[dof_y1] * sin
    delta_u2 = displacements[dof_x2] * cos + displacements[dof_y2] * sin
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

    # Debug output for the first few members
    if member_id <= 3 or member_id == 32:
        print(f"\nMember {member_id} analysis:")
        print(f"  Length = {mem_data['length_ft']:.2f} ft")
        print(f"  Node {start}: dx = {displacements[dof_x1]*12:.4f} in, dy = {displacements[dof_y1]*12:.4f} in")
        print(f"  Node {end}: dx = {displacements[dof_x2]*12:.4f} in, dy = {displacements[dof_y2]*12:.4f} in")
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
for member_id, status_info in member_status.items():
    if status_info["safety_factor"] < min_safety_factor and status_info["safety_factor"] < 100:
        members_below_min_sf.append({
            'id': member_id,
            'sf': status_info["safety_factor"],
            'stress': status_info["stress"],
            'allowable': status_info["allowable"]
        })

if members_below_min_sf:
    print(f"\n{len(members_below_min_sf)} members have safety factor below {min_safety_factor}:")
    for mem in sorted(members_below_min_sf, key=lambda x: x['sf'])[:10]:  # Show worst 10
        print(f"  Member {mem['id']}: SF = {mem['sf']:.2f}, Stress = {abs(mem['stress']):.1f} ksi")

# Create figure for the PDF - add timestamp to filenames
timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
pdf_filename = os.path.join(os.path.dirname(__file__), f"Pratt_Bridge_Analysis_{timestamp}.pdf")
text_filename = os.path.join(os.path.dirname(__file__), f"Pratt_Bridge_Analysis_{timestamp}.txt")

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
        txt_report.write("WARNING: High condition number indicates numerical instability.\n")
        txt_report.write("The structure may have:\n")
        txt_report.write("1. Duplicate nodes at the same location\n")
        txt_report.write("2. Insufficient supports (mechanism)\n")
        txt_report.write("3. Disconnected parts\n")
        txt_report.write("4. Numerical scaling issues\n\n")

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
    txt_report.write("Node | dx (inches) | dy (inches)\n")
    txt_report.write("-" * 35 + "\n")

    for node_id in sorted(node_dict.keys()):
        dof_x, dof_y = get_dof(node_id)
        dx_in = displacements[dof_x] * 12
        dy_in = displacements[dof_y] * 12
        txt_report.write(f"{node_id:4} | {dx_in:11.4f} | {dy_in:11.4f}\n")

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

    # 2. Reliability Analysis - Safety Factors plot
    fig, ax = plt.subplots(figsize=(11, 8.5))  # Changed to landscape orientation

    # Use plt.colormaps instead of cm.get_cmap to avoid deprecation warning
    cmap = plt.colormaps['RdYlGn']  # Red-Yellow-Green colormap
    norm = plt.Normalize(0.5, 2.0)

    # Plot members with color based on safety factors
    for member in members:
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        safety_factor = member_safety_factors[member_id]
        safety_factor_capped = min(safety_factor, 5.0)  # Cap extremely high values for visualization

        x1, y1, _ = node_dict[i_node]
        x2, y2, _ = node_dict[j_node]
        x = [x1, x2]
        y = [y1, y2]

        color = cmap(norm(safety_factor_capped))
        linewidth = 3  # Consistent line width

        ax.plot(x, y, color=color, linewidth=linewidth)

        # Add safety factor label at midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Format infinity as "∞"
        sf_text = "∞" if safety_factor > 100 else f"{safety_factor:.1f}"

        ax.text(mid_x, mid_y, sf_text,
                fontsize=6, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Plot nodes and supports
    for node_id, (x, y, _) in node_dict.items():
        ax.plot(x, y, 'ko', markersize=5)
        ax.text(x+1, y+1, f"{node_id}", fontsize=7)

    for support in supports:
        node_id = support['node']
        x, y, _ = node_dict[node_id]
        if support['ux'] == 1 and support['uy'] == 1:
            ax.plot(x, y-2, 'ks', markersize=8)
        elif support['ux'] == 0 and support['uy'] == 1:
            ax.plot(x, y-2, 'ko', markersize=8)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Safety Factor', shrink=0.8)
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels(['0.5', '1.0 (Limit)', '1.5', '≥ 2.0'])

    # Set plot limits and labels
    max_x = max(node_dict[n][0] for n in node_dict)
    max_y = max(node_dict[n][1] for n in node_dict)
    ax.set_xlim(-5, max_x + 15)
    ax.set_ylim(-5, max_y + 5)
    ax.set_xlabel('Length (ft)')
    ax.set_ylabel('Height (ft)')
    ax.set_title('Pratt Truss Bridge Reliability Analysis - Safety Factors', fontsize=12)
    ax.grid(True)
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])

    # Save the safety factor visualization to PDF
    pdf.savefig(fig)
    plt.close(fig)

    # 3. Member Forces Summary Tables (split across pages) with node IDs
    rows_per_page = 20  # Adjust as needed
    all_members = sorted(members, key=lambda m: m['id'])
    member_chunks = [all_members[i:i+rows_per_page] for i in range(0, len(all_members), rows_per_page)]

    for page_idx, member_chunk in enumerate(member_chunks):
        fig = plt.figure(figsize=(8.5, 11))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        ax.axis('tight')
        ax.axis('off')

        # Create summary data for this chunk
        table_data = []
        table_data.append(['Member', 'Nodes (i-j)', 'Force (kips)', 'Type'])

        for member in member_chunk:
            member_id = member['id']
            i_node = member['i_node']
            j_node = member['j_node']
            force = member_forces[member_id]

            if abs(force) < ZERO_FORCE_THRESHOLD:
                force_type = "ZERO"
            elif force > 0:
                force_type = "TENSION"
            else:
                force_type = "COMPRESSION"

            table_data.append([str(member_id), f"{i_node}-{j_node}", f"{abs(force):.2f}", force_type])

        # Create table with smaller font
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        # Add title with page number but more space at the top
        plt.title(f"Member Forces Summary (Page {page_idx+1} of {len(member_chunks)})",
                  fontsize=14, y=0.98, pad=20)

        # Add metadata
        plt.figtext(0.5, 0.03,
                    f"Pratt Bridge Analysis - Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    ha='center', fontsize=6)

        # Save the page to PDF
        pdf.savefig(fig)
        plt.close(fig)

    # 4. Member Forces plot
    fig, ax = plt.subplots(figsize=(11, 8.5))  # Changed to landscape orientation

    # Plot members with color based on tension/compression/zero
    for member in members:
        member_id = member['id']
        i_node = member['i_node']
        j_node = member['j_node']
        force = member_forces[member_id]

        # Find the coordinates for both nodes
        x1, y1, _ = node_dict[i_node]
        x2, y2, _ = node_dict[j_node]
        x = [x1, x2]
        y = [y1, y2]

        # Color: red for tension, blue for compression, black for zero/near-zero
        if abs(force) < ZERO_FORCE_THRESHOLD:
            color = 'black'  # Zero force
            linestyle = '-'
            linewidth = 1.5
        elif force > 0:
            color = 'red'  # Tension
            linestyle = '-'
            linewidth = 2 + min(5, abs(force)/100)  # Scale width with force magnitude
        else:
            color = 'blue'  # Compression
            linestyle = '-'
            linewidth = 2 + min(5, abs(force)/100)  # Scale width with force magnitude

        # Plot the member
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)

        # Add force label at midpoint - reduced font size
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, f"{abs(force):.1f}",
                fontsize=6, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Plot nodes
    for node_id, (x, y, _) in node_dict.items():
        ax.plot(x, y, 'ko', markersize=6)
        ax.text(x+1, y+1, f"{node_id}", fontsize=7)

    # Plot supports
    for support in supports:
        node_id = support['node']
        x, y, _ = node_dict[node_id]

        # Different markers for different support types
        if support['ux'] == 1 and support['uy'] == 1:
            # Pinned support - triangle
            ax.plot(x, y-2, 'ks', markersize=10)
        elif support['ux'] == 0 and support['uy'] == 1:
            # Roller support - circle
            ax.plot(x, y-2, 'ko', markersize=10)

    # Add legend
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Tension')
    blue_line = plt.Line2D([0], [0], color='blue', linewidth=3, label='Compression')
    black_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Zero Force')
    ax.legend(handles=[red_line, blue_line, black_line], loc='upper right', fontsize=8)

    # Set plot limits and labels
    ax.set_xlim(-5, max_x + 15)  # Extra space on right for labels
    ax.set_ylim(-5, max_y + 5)
    ax.set_xlabel('Length (ft)')
    ax.set_ylabel('Height (ft)')
    ax.set_title('Pratt Truss Bridge Analysis - Member Forces', fontsize=12)
    ax.grid(True)

    # Use tight_layout with adjusted parameters
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])

    # Save the member forces plot to PDF
    pdf.savefig(fig)
    plt.close(fig)

    # 5. Displacement visualization
    fig, ax = plt.subplots(figsize=(12, 9))  # Increased figure size for better visibility

    # Draw original structure in light gray
    for member in members:
        i_node = member['i_node']
        j_node = member['j_node']
        x1, y1, _ = node_dict[i_node]
        x2, y2, _ = node_dict[j_node]
        ax.plot([x1, x2], [y1, y2], color='lightgray', linestyle='-', linewidth=1, alpha=0.5)

    # Calculate max displacement for proper scaling
    max_disp_magnitude = 0
    for node_id in node_dict:
        dof_x, dof_y = get_dof(node_id)
        disp_magnitude = np.sqrt(displacements[dof_x]**2 + displacements[dof_y]**2)
        max_disp_magnitude = max(max_disp_magnitude, disp_magnitude)

    # More reasonable scale factor - aim for approximately 5-10% of bridge span
    max_span = max(node_dict[n][0] for n in node_dict) - min(node_dict[n][0] for n in node_dict)
    bridge_height = max(node_dict[n][1] for n in node_dict) - min(node_dict[n][1] for n in node_dict)
    target_max_disp = min(max_span, bridge_height) * 0.05  # More conservative scaling

    # Calculate scale factor (with protection against division by zero)
    if max_disp_magnitude > 1e-10:
        scale_factor = target_max_disp / max_disp_magnitude
        # Ensure scale factor is rounded to a clean number for display
        scale_magnitude = 10 ** int(np.log10(scale_factor))
        scale_factor = round(scale_factor / scale_magnitude) * scale_magnitude
    else:
        scale_factor = 50.0  # Default if displacements are effectively zero

    # Print debug info for node 12 (central top chord)
    if 12 in node_dict:
        dof_x12, dof_y12 = get_dof(12)
        print(f"\nDebug - Node 12 displacement:")
        print(f"  Original position: ({node_dict[12][0]}, {node_dict[12][1]})")
        print(f"  Displacement: dx={displacements[dof_x12]*12:.6f} in, dy={displacements[dof_y12]*12:.6f} in")
        print(f"  Scaled displacement: dx={displacements[dof_x12]*scale_factor:.4f} ft, dy={displacements[dof_y12]*scale_factor:.4f} ft")

    # Precalculate the max and min coordinates after displacement to ensure proper plot limits
    min_x_disp = float('inf')
    min_y_disp = float('inf')
    max_x_disp = float('-inf')
    max_y_disp = float('-inf')

    # First pass - calculate extents of displaced structure
    for node_id, (x, y, _) in node_dict.items():
        dof_x, dof_y = get_dof(node_id)
        x_new = x + displacements[dof_x] * scale_factor
        y_new = y + displacements[dof_y] * scale_factor

        min_x_disp = min(min_x_disp, x_new)
        min_y_disp = min(min_y_disp, y_new)
        max_x_disp = max(max_x_disp, x_new)
        max_y_disp = max(max_y_disp, y_new)

    # Draw displaced structure
    for member in members:
        i_node = member['i_node']
        j_node = member['j_node']
        x1, y1, _ = node_dict[i_node]
        x2, y2, _ = node_dict[j_node]

        # Get displacements
        dof_x1, dof_y1 = get_dof(i_node)
        dof_x2, dof_y2 = get_dof(j_node)

        # Apply displacements
        x1_new = x1 + displacements[dof_x1] * scale_factor
        y1_new = y1 + displacements[dof_y1] * scale_factor
        x2_new = x2 + displacements[dof_x2] * scale_factor
        y2_new = y2 + displacements[dof_y2] * scale_factor

        # Draw displaced member
        ax.plot([x1_new, x2_new], [y1_new, y2_new], color='blue', linestyle='-', linewidth=1.5)

        # Add small markers at displaced node positions for clarity
        ax.plot(x1_new, y1_new, 'b.', markersize=4)
        ax.plot(x2_new, y2_new, 'b.', markersize=4)

    # Highlight the node with maximum displacement - directly on the plot without a box
    if max_disp_node:
        x, y, _ = node_dict[max_disp_node]
        dof_x, dof_y = get_dof(max_disp_node)
        x_new = x + displacements[dof_x] * scale_factor
        y_new = y + displacements[dof_y] * scale_factor

        # Use a more visible marker directly on the plot
        ax.plot(x_new, y_new, 'ro', markersize=8)

        # Add the label directly at the node with smaller text and no box
        label_x = x_new + 0.03 * max_span
        label_y = y_new + 0.02 * max_span
        ax.text(label_x, label_y, f"Node {max_disp_node}: {max_vert_disp_in:.2f} in",
                fontsize=8, color='red', ha='left', va='bottom',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')])

    # Add legend
    orig = plt.Line2D([0], [0], color='lightgray', linestyle='-', linewidth=1, alpha=0.5, label='Original Structure')
    disp = plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=1.5, label='Displaced Structure')
    ax.legend(handles=[orig, disp], loc='upper right', fontsize=8)

    # Set plot limits to ensure complete structure visibility, with additional padding
    min_x = min(node_dict[n][0] for n in node_dict)
    min_y = min(node_dict[n][1] for n in node_dict)
    max_x = max(node_dict[n][0] for n in node_dict)
    max_y = max(node_dict[n][1] for n in node_dict)

    # Ensure we include both original and displaced structures with padding
    plot_min_x = min(min_x, min_x_disp) - 0.1 * max_span
    plot_min_y = min(min_y, min_y_disp) - 0.15 * bridge_height
    plot_max_x = max(max_x, max_x_disp) + 0.1 * max_span
    plot_max_y = max(max_y, max_y_disp) + 0.1 * bridge_height

    ax.set_xlim(plot_min_x, plot_max_x)
    ax.set_ylim(plot_min_y, plot_max_y)
    ax.set_xlabel('Length (ft)')
    ax.set_ylabel('Height (ft)')

    # Format scale factor nicely
    if scale_factor >= 1000:
        scale_factor_text = f"{scale_factor/1000:.1f}k"
    elif scale_factor >= 100:
        scale_factor_text = f"{int(scale_factor)}"
    else:
        scale_factor_text = f"{scale_factor:.1f}"

    ax.set_title(f'Pratt Truss Bridge - Displacement Analysis (Scale Factor: {scale_factor_text}×)', fontsize=12)

    # Add note about displacement scale
    plt.figtext(0.5, 0.01,
                f"Note: Displacements are scaled by a factor of {scale_factor_text} for visibility. "
                f"Actual maximum displacement: {max_vert_disp_in:.4f} inches.",
                ha='center', fontsize=8)

    ax.grid(True)
    plt.tight_layout(pad=2.0, rect=[0, 0.02, 1, 0.98])

    # Save the displacement page to PDF
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
    print("1. The duplicate nodes 6 and 13 at (100, 0) should be merged into a single node")
    print("2. Update member connectivity to reference the merged node")
    print("3. Ensure all supports are properly defined")
    print("4. Check for any disconnected parts of the structure")

# Display the plot on screen
plt.show()
