import os
from os import path
import sys
import re  # For regex parsing of unit strings

# Add pint for units handling
import pint
# Set up a unit registry with kN as the preferred force unit
ureg = pint.UnitRegistry()
ureg.default_system = 'mks'  # meter-kilogram-second
Q_ = ureg.Quantity  # Shorthand for creating quantities

# Define the standard units for the program
FORCE_UNIT = 'kN'
LENGTH_UNIT = 'm'
MOMENT_UNIT = 'kN*m'
PRESSURE_UNIT = 'kN/m^2'
DISTRIBUTED_LOAD_UNIT = 'kN/m'

import numpy as np
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal')

# Add custom formatters for different numeric types
def format_with_dots(x): return '.' if abs(x) < 1e-10 else f"{x:.4g}"
def format_double(x): return '.' if abs(x) < 1e-10 else f"{x:.8g}"  # More precision for doubles

# np.set_printoptions(formatter={
#     'float': format_with_dots,     # For float32 and generic floats
#     'float_kind': format_with_dots,  # For all floating point types
#     'float64': format_double       # Specifically for float64 (double precision)
# })

from pyMAOS.plot_structure import plot_structure_vtk
 
from contextlib import redirect_stdout

from context import pyMAOS

from pyMAOS.nodes import R2Node

from pyMAOS.Frame import R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo

default_scaling = {
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    }

# Update the regex pattern in parse_value_with_units function
def parse_value_with_units(value_string):
    """Parse a string that may contain a value with units without spaces between"""
    # Match pattern: [numeric value][units]
    # The numeric part can include scientific notation like 30.0e6
    # Units part starts at the first non-numeric, non-exponent character
    match = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(.*)', value_string.strip())
    
    if match:
        value_str, unit_str = match.groups()
        value = float(value_str)
        
        if unit_str and unit_str.strip():
            try:
                # Create quantity with units
                value_with_units = Q_(value, unit_str)
                return value_with_units
            except:
                print(f"Warning: Could not parse unit '{unit_str}', treating as dimensionless")
                return value
        return value
    
    # If no match, try to evaluate as a simple numeric expression
    try:
        return float(eval(value_string))
    except:
        raise ValueError(f"Could not parse value: {value_string}")

def load_frame_from_text(filename):
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
    with open(filename, 'r', encoding='ascii') as file:
        lines = [line.strip().split('#')[0].strip() for line in file]
        lines = [line for line in lines if line]
    # print(lines)

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
        rz = parts[3]
 
        nodes_dict[node_id].restraints = [rx, ry, rz]
        print(f"Node {node_id} supports: rx={rx}, ry={ry}, rz={rz}")
        line_idx += 1
    print(f"Read {num_supports} supports for {len(nodes_dict)} nodes.")
  
    
    # 3. Read materials with unit support
    num_materials = int(lines[line_idx]); line_idx += 1
    
    materials_dict = {}
    for i in range(num_materials):
        # Get the raw line
        line = lines[line_idx].strip()
        
        # Parse the value with units
        value_with_units = parse_value_with_units(line)
        
        # Convert to kN/m² instead of Pa
        if isinstance(value_with_units, pint.Quantity):
            try:
                e_value = value_with_units.to(PRESSURE_UNIT).magnitude
                print(f"Converted material property from {value_with_units} to {e_value} {PRESSURE_UNIT}")
            except:
                print(f"Warning: Could not convert {value_with_units} to {PRESSURE_UNIT}, using raw value")
                e_value = value_with_units.magnitude
        else:
            e_value = value_with_units
            print(f"No units specified for material property, using value: {e_value}")
            
        material = Material(uid=i + 1, E=e_value)
        materials_dict[i + 1] = material
        line_idx += 1
    
    # 4. Read cross-sections with unit support
    num_sections = int(lines[line_idx]); line_idx += 1
    
    sections_dict = {}
    for i in range(num_sections):
        # Split by whitespace to get the two property values
        parts = lines[line_idx].split()
        
        # Parse each part with potential units
        if len(parts) >= 2:
            area_with_units = parse_value_with_units(parts[0])
            ixx_with_units = parse_value_with_units(parts[1])
            
            # Convert Area to m² if it has units
            if isinstance(area_with_units, pint.Quantity):
                try:
                    area = area_with_units.to('m^2').magnitude
                    print(f"Converted section area from {area_with_units} to {area} m²")
                except:
                    print(f"Warning: Could not convert {area_with_units} to m², using raw value")
                    area = area_with_units.magnitude
            else:
                area = area_with_units
                
            # Convert Ixx to m⁴ if it has units
            if isinstance(ixx_with_units, pint.Quantity):
                try:
                    ixx = ixx_with_units.to('m^4').magnitude
                    print(f"Converted section inertia from {ixx_with_units} to {ixx} m⁴")
                except:
                    print(f"Warning: Could not convert {ixx_with_units} to m⁴, using raw value")
                    ixx = ixx_with_units.magnitude
            else:
                ixx = ixx_with_units
        else:
            # Fallback for old format
            area = float(eval(parts[0]))
            ixx = float(eval(parts[1]) if len(parts) > 1 else 0)
            
        section = Section(uid=i + 1, Area=area, Ixx=ixx)
        sections_dict[i + 1] = section
        line_idx += 1
    
    # 5. Read elements
    num_elements = int(lines[line_idx]); line_idx += 1
    
    element_list = []
    elements_dict = {}  # Dictionary to store elements by UID for member loads
    for i in range(num_elements):
        parts = [int(x.strip()) for x in lines[line_idx].split(',')]
        i_node = parts[0]
        j_node = parts[1]
        mat_id = parts[2]
        sec_id = parts[3]
        
        # Use R2Frame for frame elements
        element = R2Frame(
            uid=i + 1,
            inode=nodes_dict[i_node],
            jnode=nodes_dict[j_node],
            material=materials_dict[mat_id],
            section=sections_dict[sec_id]
        )
        element_list.append(element)
        elements_dict[i + 1] = element
        line_idx += 1
    print(f"Read {len(element_list)} elements.")

    # 6. Read joint loads with unit support
    num_loads = int(lines[line_idx]); line_idx += 1
    
    for i in range(num_loads):
        parts = [x.strip() for x in lines[line_idx].split(',')]
        node_id = int(parts[0])
        
        # Parse forces with possible units
        fx_with_units = parse_value_with_units(parts[1])
        fy_with_units = parse_value_with_units(parts[2])
        fz_with_units = parse_value_with_units(parts[3]) if len(parts) > 3 else 0
        
        # Convert to kN instead of N
        fx = fx_with_units.to(FORCE_UNIT).magnitude if isinstance(fx_with_units, pint.Quantity) else fx_with_units
        fy = fy_with_units.to(FORCE_UNIT).magnitude if isinstance(fy_with_units, pint.Quantity) else fy_with_units
        
        # For moment (fz), convert to kN·m if it has units
        if isinstance(fz_with_units, pint.Quantity):
            fz = fz_with_units.to(MOMENT_UNIT).magnitude
        else:
            fz = fz_with_units
        
        print(f"Node {node_id} load: Fx={fx}{FORCE_UNIT}, Fy={fy}{FORCE_UNIT}, Mz={fz}{MOMENT_UNIT}")
        nodes_dict[node_id].add_nodal_load(fx, fy, fz, "D")
        line_idx += 1
    print(f"Read {num_loads} joint loads for {len(nodes_dict)} nodes.")

    # Print remaining lines (useful for debugging)
    print("\nRemaining lines from current position:")
    for i in range(line_idx, len(lines)):
        print(f"Line {i}: {lines[i]}")

    # 7. Read member loads if available
    if line_idx < len(lines):
        num_member_loads = int(lines[line_idx]); line_idx += 1
        print(f"\nProcessing {num_member_loads} member loads:")
        
        # Import the necessary load classes
        from pyMAOS.loading import R2_Point_Load, R2_Linear_Load, R2_Axial_Load, R2_Axial_Linear_Load, R2_Point_Moment
        
        for i in range(num_member_loads):
            if line_idx >= len(lines):
                break
                
            parts = [x.strip() for x in lines[line_idx].split(',')]
            element_id = int(parts[0])
            
            if element_id not in elements_dict:
                print(f"Warning: Member load specified for non-existent element {element_id}")
                line_idx += 1
                continue
                
            element = elements_dict[element_id]
            load_type = int(parts[1])
            
            # Map load type to appropriate load class based on the corrected mapping
            try:
                if load_type == 1:  # R2_Point_Load - Concentrated transverse/shear load
                    p_with_units = parse_value_with_units(parts[2])
                    p = p_with_units.to(FORCE_UNIT).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units
                    a = float(parts[4])  # Position
                    print(f"  Element {element_id}: Point load (shear) p={p}{FORCE_UNIT}, position={a}")
                    element.add_point_load(p, a, "D")

                elif load_type == 2:  # R2_Point_Moment - Concentrated moment
                    M_with_units = parse_value_with_units(parts[2])
                    M = M_with_units.to(MOMENT_UNIT).magnitude if isinstance(M_with_units, pint.Quantity) else M_with_units
                    a = float(parts[3])  # Position
                    print(f"  Element {element_id}: Point moment M={M}{MOMENT_UNIT}, position={a}")
                    element.add_point_moment(M, a, "D")
                    
                elif load_type == 3:  # R2_Linear_Load - Distributed transverse load
                    w1_with_units = parse_value_with_units(parts[2])
                    w1 = w1_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units, pint.Quantity) else w1_with_units

                    if len(parts) >= 4:
                        w2_with_units = parse_value_with_units(parts[3])
                        w2 = w2_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units, pint.Quantity) else w2_with_units
                    else:
                        w2 = w1
                    
                    # Handle based on number of parameters
                    if len(parts) >= 5:  # At least w1, w2, a, b
                       a = float(parts[4]) if len(parts) > 5 else float(parts[3])
                       b = float(parts[5]) if len(parts) > 5 else element.length
                       print(f"  Element {element_id}: Distributed linear load w1={w1}, w2={w2} a={a}, b={b}")

                        # Apply as uniform distributed load with full member length
                       # load = R2_Linear_Load(w1, w2, a, b, element, "D")
                       element.add_distributed_load(w1, w2, a, b, "D")

                    elif len(parts) == 4:  # Just w1 and one extra parameter
                        # Assume w2=w1 (uniform load) from a=parts[3] to b=member.length
                        a = float(parts[3])
                        b = element.length
                        print(f"  Element {element_id}: Linear load w1={w1}, w2={w1}, a={a}, b={b}")
                        # load = R2_Linear_Load(w1, w1, a, b, element, "D")
                        element.add_distributed_load(w1, w1, a, b, "D")
                        
                    else:
                        # Basic case: uniform load over full member
                        print(f"  Element {element_id}: Linear load w1={w1}, w2={w1}, full member")
                        # load = R2_Linear_Load(w1, w1, 0.0, element.length, element, "D")
                        element.add_linear_load(w1, w1, 0.0, element.length,"D")
                    
                elif load_type == 4:  # R2_Axial_Linear_Load - Distributed axial load
                    w1 = float(parts[2])  # Start intensity
                    w2 = w1  # Default: constant intensity
                    a = 0.0  # Default: start of member
                    b = element.length  # Default: end of member
                    
                    if len(parts) > 3:
                        w2 = float(parts[3])
                    if len(parts) > 4:
                        a = float(parts[4])
                    if len(parts) > 5:
                        b = float(parts[5])
                    
                    print(f"  Element {element_id}: Axial linear load w1={w1}, w2={w2}, a={a}, b={b}")
                    load = R2_Axial_Linear_Load(w1, w2, a, b, element, "D")
                    element.add_distributed_load(w1, w2, a, b,  "D", direction="xx")
                    
                elif load_type == 5:  # R2_Axial_Load - Concentrated axial load
                    # For case in figure_6_19_input.txt: Element 3, type 5, 20.0, 129.24
                    p = float(parts[2])  # Force value
                    a = float(parts[4])  # Position
                    print(f"  Element {element_id}: Axial load p={p}, position={a}")
                    load = R2_Axial_Load(p, a, element, "D")
                    element.add_point_load(p, a,"D", direction="xx")
                    
                else:
                    print(f"  Warning: Unknown load type {load_type}, skipping")
                    
            except Exception as e:
                print(f"  Error applying load to element {element_id}: {e}")
            
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
     
    # Print node load information
    print("\n\n--- Node Load Summary ---")
    for node in node_list:
        if node.loads:
            print(f"Node {node.uid} loads:")
            for case, load in node.loads.items():
                print(f"  Case {case}: Fx={load[0]}, Fy={load[1]}, Mz={load[2]}")
    # plot structure
    plot_structure_vtk(node_list, element_list, scaling=default_scaling)

    return node_list, element_list

if __name__ == "__main__":
     working_dir='example_6_7'; input_File='example_6_7.txt'
     # working_dir='figure_6_19'; input_File='figure_6_19_input.txt'
     node_list, element_list = load_frame_from_text(os.path.join(working_dir, input_File))
     print(f"Total nodes: {len(node_list)}")
     print(f"Total elements: {len(element_list)}")

     Structure = R2Struct.R2Structure(node_list, element_list)
     FM = Structure.freedom_map(); print(FM)
     
     # Pass output_dir parameter to Kstructure
     K = Structure.Kstructure(FM, output_dir=working_dir)
     np.save(os.path.join(working_dir, 'K.npy'), K)  # Save K matrix to file
     np.savetxt(os.path.join(working_dir, 'K.txt'), K) 
     print(f"Global Stiffness Matrix K:\n{K}")
     
     # Fix the LoadCombo initialization with proper parameters
     loadcombo = LoadCombo("D", {"D": 1.0}, ["D"], False, "SLS")
     print("Solving linear static problem...")
     U = Structure.solve_linear_static(loadcombo, output_dir=working_dir, verbose=True)
     print(f"Displacements U:\n{U}")

     # Save displacement results
     np.save(os.path.join(working_dir, 'U.npy'), U)
     np.savetxt(os.path.join(working_dir, 'U.txt'), U)

     Structure.plot_loadcombos_vtk(loadcombos=None, scaling=default_scaling)
      
     # Pause the program before exiting
     print("\n\nAnalysis complete. Press Enter to exit...")