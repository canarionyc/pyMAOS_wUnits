import os
from os import path
import sys


import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=np.nan, threshold=sys.maxsize) # type: ignore

from pyMAOS.R2Node import R2Node
from pyMAOS.R2Truss import R2Truss
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo


def read_truss_from_text(filename):
    """
    Reads a structural model from example_4_2_input.txt format
    
    Parameters
    ----------
    filename : str
        Path to the input file
        
    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis
    """
    with open(filename, 'r') as file:
        lines = [line.strip().split('#')[0].strip() for line in file]
        lines = [line for line in lines if line]
    
    line_idx = 0
    
    # 1. Read nodes
    num_nodes = int(lines[line_idx]); line_idx += 1
    
    nodes_dict = {}
    for i in range(num_nodes):
        coords = [float(x.strip()) for x in lines[line_idx].split(',')]
        node_id = i + 1  # 1-based indexing
        node = R2Node(node_id, coords[0], coords[1])
        nodes_dict[node_id] = node
        line_idx += 1
    print(f"Read {len(nodes_dict)} nodes."); print(nodes_dict)

    # 2. Read supports
    num_supports = int(lines[line_idx]); line_idx += 1
    
    for i in range(num_supports):
        parts = [int(x.strip()) for x in lines[line_idx].split(',')]
        node_id = parts[0]
        rx = parts[1]
        ry = parts[2]
        rz = 1  # For truss nodes, set rotational restraint to 1    
        nodes_dict[node_id].restraints = [rx, ry, rz]
        print(f"Node {node_id} supports: rx={rx}, ry={ry}, rz={rz}")
        line_idx += 1
    print(f"Read {num_supports} supports for {len(nodes_dict)} nodes.")
    
    # Set all other nodes to have rz=0 for truss compatibility
    for node_id, node in nodes_dict.items():
        node.restraints[2] = 1
        print(f"Node {node_id} restraints set to: {node.restraints}")
    
    # 3. Read materials
    num_materials = int(lines[line_idx]); line_idx += 1
    
    materials_dict = {}
    for i in range(num_materials):
        props = [float(x) for x in lines[line_idx].split()]
        material = Material(uid=i + 1, E=props[0])
        materials_dict[i + 1] = material
        line_idx += 1
    
    # 4. Read sections
    num_sections = int(lines[line_idx]); line_idx += 1
    
    sections_dict = {}
    for i in range(num_sections):
        area = float(lines[line_idx])
        section = Section(uid=i + 1, Area=area)  # Area and Ixx (using area as Ixx)
        sections_dict[i + 1] = section
        line_idx += 1
    
    # 5. Read elements
    num_elements = int(lines[line_idx]); line_idx += 1
    
    element_list = []
    for i in range(num_elements):
        parts = [int(x.strip()) for x in lines[line_idx].split(',')]
        i_node = parts[0]
        j_node = parts[1]
        mat_id = parts[2]
        sec_id = parts[3]
        
        # Use R2Truss instead of R2Frame
        element = R2Truss(
            uid=i + 1,
            inode=nodes_dict[i_node],
            jnode=nodes_dict[j_node],
            material=materials_dict[mat_id],
            section=sections_dict[sec_id]
        )
        element_list.append(element)
        line_idx += 1
    print(f"Read {len(element_list)} elements.")

    # 6. Read loads
    num_loads = int(lines[line_idx]); line_idx += 1
    
    for i in range(num_loads):
        parts = [x.strip() for x in lines[line_idx].split(',')]
        node_id = int(parts[0])
        fx = float(parts[1])
        fy = float(parts[2])
        fz = float(parts[3])
        nodes_dict[node_id].add_nodal_load(fx, fy, fz, "D")
        line_idx += 1
    
    node_list = [nodes_dict[uid] for uid in sorted(nodes_dict)]

       # Print node restraints
    print("\n\n--- Node Restraints Summary ---")
    print("Node ID  |  Ux  |  Uy  |  Rz")
    print("-" * 30)
    for node in node_list:
        rx, ry, rz = node.restraints
        rx_status = "Fixed" if rx == 1 else "Free"
        ry_status = "Fixed" if ry == 1 else "Free"
        rz_status = "Fixed" if rz == 1 else "Free"
        print(f"Node {node.uid:2d}  |  {rx_status:4s} |  {ry_status:4s} |  {rz_status:4s}")
     
    # Print load information
    print("\n\n--- Load Summary ---")
    for node in node_list:
        if node.loads:
            print(f"Node {node.uid} loads:")
            for case, load in node.loads.items():
                print(f"  Case {case}: Fx={load[0]}, Fy={load[1]}, Mz={load[2]}")
 
    return node_list, element_list

if __name__ == "__main__":
     node_list, element_list = read_truss_from_text('example_4_2_input/example_4_2_input.txt')
     print(f"Total nodes: {len(node_list)}")
     print(f"Total elements: {len(element_list)}")

     Structure = R2Struct.R2Structure(node_list, element_list)
     FM = Structure.freedom_map(); print(FM)

     # Create output directory if it doesn't exist
     os.makedirs('example_4_2_output', exist_ok=True)
     
     # Pass output_dir parameter to Kstructure
     K = Structure.Kstructure(FM, output_dir='example_4_2_output')
     np.save(os.path.join('example_4_2_output', 'K.npy'), K)  # Save K matrix to file
     np.savetxt(os.path.join('example_4_2_output', 'K.txt'), K) 
     print(f"Global Stiffness Matrix K:\n{K}")
     
     # Fix the LoadCombo initialization with proper parameters
     loadcombo = LoadCombo("D", {"D": 1.0}, ["D"], False, "SLS")
     print("Solving linear static problem...")
     U = Structure.solve_linear_static(loadcombo, output_dir='example_4_2_output', verbose=True)
     print(f"Displacements U:\n{U}")

     # Save displacement results
     np.save(os.path.join('example_4_2_output', 'U.npy'), U)
     np.savetxt(os.path.join('example_4_2_output', 'U.txt'), U) 

      
     # Pause the program before exiting
     print("\n\nAnalysis complete. Press Enter to exit...")
     input()  # Wait for user to press Enter