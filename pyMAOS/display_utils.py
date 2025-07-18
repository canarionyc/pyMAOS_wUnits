def display_node_displacement_in_units(node_displacement_vector, node_uid=None, length_unit='m', load_combo_name=None):
    """
    Displays a node displacement vector with appropriate units for each component.
    """
    import numpy as np
    from pyMAOS import units
    
    # Convert to numpy array if it's not already
    if isinstance(node_displacement_vector, (list, tuple)):
        node_displacement_vector = np.array(node_displacement_vector)
    elif isinstance(node_displacement_vector, np.matrix):
        node_displacement_vector = np.array(node_displacement_vector).flatten()
    
    # Define the DOF labels and units
    dof_labels = ["Ux", "Uy", "θz"]
    units_list = [length_unit, length_unit, "rad"]
    
    # Convert values from internal units to display units
    converted_values = node_displacement_vector.copy()
    # Convert translations (index 0 and 1) from meters to specified length unit
    converted_values[0] = units.Q_(node_displacement_vector[0], 'm').to(length_unit).magnitude
    converted_values[1] = units.Q_(node_displacement_vector[1], 'm').to(length_unit).magnitude
    # Rotation (index 2) remains in radians
    
    # Print header information
    header = "\nNode Displacement Vector"
    if node_uid is not None:
        header += f" for Node {node_uid}"
    if load_combo_name is not None:
        header += f" under '{load_combo_name}'"
    print(header)
    
    # Print column headers
    print(f"{'DOF':10} {'Value':15} {'Unit':8} {'Description':20}")
    print("-" * 55)
    
    # Print each component with its unit
    descriptions = ["X-Translation", "Y-Translation", "Z-Rotation"]
    for i, (dof, unit, value, desc) in enumerate(zip(dof_labels, units_list, converted_values, descriptions)):
        # Format the value with appropriate precision based on magnitude
        if abs(value) < 1e-10:
            formatted_value = "0"
        elif abs(value) < 0.001:
            formatted_value = f"{value:.8e}"
        else:
            formatted_value = f"{value:.6g}"
            
        print(f"{dof:10} {formatted_value:15} {unit:8} {desc:20}")
    
    print("-" * 55)
    
    # Return the original vector for potential chaining
    return node_displacement_vector

def display_node_load_vector_in_units(load_vector, node_uid=None, force_unit='N', length_unit='m', load_combo_name=None):
    """
    Displays a nodal force vector with appropriate units for each component.
    
    Parameters
    ----------
    load_vector : numpy.ndarray or list
        3-element vector containing nodal forces [Fx, Fy, Mz]
    node_uid : int or str, optional
        Node ID for display purposes
    force_unit : str, optional
        Force unit for display (default: 'N')
    length_unit : str, optional
        Length unit for moments (default: 'm')
    load_combo_name : str, optional
        Load combination name for display purposes
    
    Returns
    -------
    numpy.ndarray
        The original force vector for potential chaining
    """
    import numpy as np
    from pyMAOS import units
    
    # Convert to numpy array if it's not already
    if isinstance(load_vector, (list, tuple)):
        load_vector = np.array(load_vector)
    elif isinstance(load_vector, np.matrix):
        load_vector = np.array(load_vector).flatten()
    
    # Define the DOF labels and units
    dof_labels = ["Fx", "Fy", "Mz"]
    units_list = [force_unit, force_unit, f"{force_unit}·{length_unit}"]
    
    # Convert values from internal units to display units
    converted_values = load_vector.copy()
    # Convert forces (index 0 and 1) from N to specified force unit
    converted_values[0] = units.Q_(load_vector[0], 'N').to(force_unit).magnitude
    converted_values[1] = units.Q_(load_vector[1], 'N').to(force_unit).magnitude
    # Convert moment (index 2) from N*m to specified force*length unit
    moment_unit = f"{force_unit}*{length_unit}"
    converted_values[2] = units.Q_(load_vector[2], 'N*m').to(moment_unit).magnitude
    
    # Print header information
    header = "\nNodal Force Vector"
    if node_uid is not None:
        header += f" for Node {node_uid}"
    if load_combo_name is not None:
        header += f" under '{load_combo_name}'"
    print(header)
    
    # Print column headers
    print(f"{'DOF':10} {'Value':15} {'Unit':12} {'Description':20}")
    print("-" * 60)
    
    # Print each component with its unit
    descriptions = ["X-Direction Force", "Y-Direction Force", "Z-Direction Moment"]
    for i, (dof, unit, value, desc) in enumerate(zip(dof_labels, units_list, converted_values, descriptions)):
        # Format the value with appropriate precision based on magnitude
        if abs(value) < 1e-10:
            formatted_value = "0"
        elif abs(value) < 0.001:
            formatted_value = f"{value:.8e}"
        else:
            formatted_value = f"{value:.6g}"
            
        print(f"{dof:10} {formatted_value:15} {unit:12} {desc:20}")
    
    print("-" * 60)
    
    # Return the original vector for potential chaining
    return load_vector

# def display_member_displacement_in_units(member_displacement_vector, force_unit='kN', length_unit='m', is_local=False, element_uid=None, load_combo_name=None):
#     """
#     Displays a displacement vector with appropriate units for each component.
    
#     Parameters
#     ----------
#     displacement_vector : numpy.ndarray or numpy.matrix
#         6-element vector containing nodal displacements
#     force_unit : str, optional
#         Force unit for reference (default: 'kN')
#     length_unit : str, optional
#         Length unit for displacements (default: 'm')
#     is_local : bool, optional
#         True if the vector is in local coordinates (default: False)
#     element_uid : int or str, optional
#         Element ID for display purposes
#     load_combo_name : str, optional
#         Load combination name for display purposes
#     """
#     import numpy as np
    
#     # Convert to numpy array if it's a matrix
#     if isinstance(member_displacement_vector, np.matrix):
#         member_displacement_vector = np.array(member_displacement_vector).flatten()
    
#     # Define the DOF labels and units
#     coordinate_system = "Local" if is_local else "Global"
#     dof_labels = [
#         f"u_i_{'x' if is_local else 'X'}", 
#         f"u_i_{'y' if is_local else 'Y'}", 
#         f"θ_i_z",
#         f"u_j_{'x' if is_local else 'X'}", 
#         f"u_j_{'y' if is_local else 'Y'}", 
#         f"θ_j_z"
#     ]
    
#     units = [
#         length_unit,   # i-node x translation
#         length_unit,   # i-node y translation
#         "rad",         # i-node z rotation
#         length_unit,   # j-node x translation
#         length_unit,   # j-node y translation
#         "rad"          # j-node z rotation
#     ]
    
#     # Print header information
#     header = f"\n{coordinate_system} Displacement Vector"
#     if element_uid is not None:
#         header += f" for Element {element_uid}"
#     if load_combo_name is not None:
#         header += f" under '{load_combo_name}'"
#     print(header)
    
#     # Print column headers
#     print(f"{'DOF':10} {'Value':15} {'Unit':8} {'Description':20}")
#     print("-" * 55)
    
#     # Print each component with its unit
#     for i, (dof, unit, value) in enumerate(zip(dof_labels, units, member_displacement_vector)):
#         # Format the value with appropriate precision based on magnitude
#         if abs(value) < 1e-10:
#             formatted_value = "0"
#         elif abs(value) < 0.001:
#             formatted_value = f"{value:.8e}"
#         else:
#             formatted_value = f"{value:.6g}"
            
#         # Determine description
#         if i % 3 == 0:  # x/X translation
#             description = "X-Translation" if not is_local else "Axial Translation"
#         elif i % 3 == 1:  # y/Y translation
#             description = "Y-Translation" if not is_local else "Transverse Translation"
#         else:  # z rotation
#             description = "Z-Rotation"
            
#         # Add node identifier
#         if i < 3:
#             description += " (i-node)"
#         else:
#             description += " (j-node)"
            
#         print(f"{dof:10} {formatted_value:15} {unit:8} {description:20}")
    
#     print("-" * 55)
    
#     # Return the original vector for potential chaining
#     return member_displacement_vector


def display_stiffness_matrix_in_units(k_matrix, force_unit='kN', length_unit='m', return_matrix=False):
    """
    Converts and displays a 6x6 local stiffness matrix from internal units (N, m) to specified display units,
    with units shown for each cell.
    
    Parameters
    ----------
    k_matrix : numpy.matrix or numpy.ndarray
        6x6 local stiffness matrix in internal units (N, m)
    force_unit : str, optional
        Target force unit (default: 'kN')
    length_unit : str, optional
        Target length unit (default: 'm')
    return_matrix : bool, optional
        If True, returns the converted matrix (default: False)
    
    Returns
    -------
    numpy.matrix or None
        Converted stiffness matrix if return_matrix is True, otherwise None
    """
    try:
        import numpy as np
        from pint import UnitRegistry
        
        # Create unit registry
        ureg = UnitRegistry()
        
        # Create conversion factors
        force_factor = ureg.parse_expression(f"1 N").to(force_unit).magnitude
        length_factor = ureg.parse_expression(f"1 m").to(length_unit).magnitude
        
        # Create converted matrix
        k_display = np.matrix(k_matrix.copy())
        
        # Create a matrix to store the units for each cell
        unit_matrix = np.empty((6, 6), dtype=object)
        
        # Default unit is force/length
        f_over_l = f"{force_unit}/{length_unit}"
        f_times_l = f"{force_unit}·{length_unit}"
        
        # Fill the unit matrix based on cell physical meaning
        for i in range(6):
            for j in range(6):
                if (i in [0, 3] and j in [0, 3]) or (i in [1, 4] and j in [1, 4]):
                    # Axial and shear stiffness: force/length
                    unit_matrix[i, j] = f_over_l
                elif ((i in [1, 4] and j in [2, 5]) or 
                      (i in [2, 5] and j in [1, 4])):
                    # Moment-shear coupling: force
                    unit_matrix[i, j] = force_unit
                elif i in [2, 5] and j in [2, 5]:
                    # Moment stiffness: force*length
                    unit_matrix[i, j] = f_times_l
                else:
                    unit_matrix[i, j] = "-"
        
        # Apply conversion factors by stiffness type to preserve mathematical relationships
        
        # Axial terms (EA/L): Convert as force/length
        axial_factor = force_factor / length_factor
        k_display[0,0] *= axial_factor
        k_display[0,3] *= axial_factor
        k_display[3,0] *= axial_factor
        k_display[3,3] *= axial_factor
        
        # Shear terms (12EI/L³): Convert as force/length
        shear_factor = force_factor / length_factor
        k_display[1,1] *= shear_factor
        k_display[1,4] *= shear_factor
        k_display[4,1] *= shear_factor
        k_display[4,4] *= shear_factor
        
        # Moment-shear coupling terms (6EI/L²): Convert as force
        coupling_factor = force_factor
        k_display[1,2] *= coupling_factor
        k_display[1,5] *= coupling_factor
        k_display[2,1] *= coupling_factor
        k_display[2,4] *= coupling_factor
        k_display[4,2] *= coupling_factor
        k_display[4,5] *= coupling_factor
        k_display[5,1] *= coupling_factor
        k_display[5,4] *= coupling_factor
        
        # Moment terms (4EI/L or 2EI/L): Convert as force*length
        moment_factor = force_factor * length_factor
        k_display[2,2] *= moment_factor
        k_display[2,5] *= moment_factor
        k_display[5,2] *= moment_factor
        k_display[5,5] *= moment_factor
        
        # Print the stiffness matrix with units
        print(f"\nStiffness matrix in {force_unit}, {length_unit} units:\n")
        
        # First display the numeric matrix
        print("NUMERIC VALUES:")
        for i in range(6):
            row = [f"{k_display[i,j]:12.4g}" for j in range(6)]
            print("  " + " ".join(row))
        
        # Then display the units matrix
        print("\nUNITS FOR EACH CELL:")
        dof_labels = ["Fx_i", "Fy_i", "Mz_i", "Fx_j", "Fy_j", "Mz_j"]
        
        # Print column headers
        print("        " + "".join([f"{dof:12s}" for dof in dof_labels]))
        
        # Print each row with row label
        for i in range(6):
            row = [f"{unit_matrix[i,j]:12s}" for j in range(6)]
            print(f"{dof_labels[i]:8s}" + "".join(row))
        
        if return_matrix:
            return k_display
    except Exception as e:
        import sys
        print(f"Error displaying stiffness matrix in units: {e}", file=sys.stderr)
        if return_matrix:
            return None


if __name__ == "__main__":
    # Example usage:
    import numpy as np

    # Some example 6x6 stiffness matrix in SI units (N, m)
    k_matrix = np.matrix([
        [1e6, 0, 0, -1e6, 0, 0],
        [0, 1.2e5, 6e4, 0, -1.2e5, 6e4],
        [0, 6e4, 4e4, 0, -6e4, 2e4],
        [-1e6, 0, 0, 1e6, 0, 0],
        [0, -1.2e5, -6e4, 0, 1.2e5, -6e4],
        [0, 6e4, 2e4, 0, -6e4, 4e4]
    ])

    # Display in kips and inches
    display_stiffness_matrix_in_units(k_matrix, force_unit='kip', length_unit='inch')

    # Display in kN and mm and get the converted matrix
    k_converted = display_stiffness_matrix_in_units(k_matrix, force_unit='kN', length_unit='mm', return_matrix=True)