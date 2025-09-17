def load_frame_from_text(filename):
    """
    Reads a structural model from input text format
    
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
        lines = [line.strip() for line in file]
        
    # Process JSON units if present
    if lines[0].strip().startswith('{'):
        if update_units_from_json(lines[0]):
            # Remove the JSON line after processing
            lines.pop(0)
    
    # Clean up lines and remove comments
    lines = [line.split('#')[0].strip() for line in lines]
    lines = [line for line in lines if line]
    
    line_idx = 0
    
    # 1. Read nodes
    num_nodes = int(lines[line_idx]); line_idx += 1
    
    nodes_dict = {}
    for i in range(num_nodes):
        coord_strings = lines[line_idx].split(',')
        # Parse coordinates with potential units
        x = parse_value_with_units(coord_strings[0].strip())
        y = parse_value_with_units(coord_strings[1].strip())
        
        # Convert to meters if units are specified
        x_meters = x.to('m').magnitude if isinstance(x, pint.Quantity) else x
        y_meters = y.to('m').magnitude if isinstance(y, pint.Quantity) else y
        
        node_id = i + 1  # 1-based indexing
        node = R2Node(node_id, x_meters, y_meters)
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
        
        if isinstance(value_with_units, pint.Quantity):
            try:
                # Try to convert to Pascal directly - most reliable approach
                e_value = value_with_units.to('Pa').magnitude
                
                # Also display in the user's preferred units for feedback
                display_value = value_with_units.to(PRESSURE_UNIT).magnitude if PRESSURE_UNIT != 'Pa' else e_value
                
                print(f"Converted material property from {value_with_units} to {e_value} Pa")
                print(f"  (Equivalent to {display_value} {PRESSURE_UNIT} in display units)")
            except Exception as e:
                # If Pa conversion fails, try the expanded form as fallback
                try:
                    e_value = value_with_units.to(INTERNAL_PRESSURE_UNIT_EXPANDED).magnitude
                    print(f"Converted material property from {value_with_units} to {e_value} Pa")
                except:
                    print(f"Warning: Could not convert {value_with_units} to Pascal, using raw value")
                    e_value = value_with_units.magnitude
        else:
            e_value = value_with_units
            print(f"No units specified for material property, assuming {e_value} Pa")
            
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
                    area = area_with_units.magnitude
                    print(f"Warning: Could not convert {area_with_units} to m², using raw value")
            else:
                area = area_with_units
                
            # Convert Ixx to m⁴ if it has units
            if isinstance(ixx_with_units, pint.Quantity):
                try:
                    ixx = ixx_with_units.to('m^4').magnitude
                    print(f"Converted section inertia from {ixx_with_units} to {ixx} m⁴")
                except:
                    ixx = ixx_with_units.magnitude
                    print(f"Warning: Could not convert {ixx_with_units} to m⁴, using raw value")
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
                
            line = lines[line_idx]
            
            # Check if we're dealing with a JSON format
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    # Parse the JSON string
                    load_data = json.loads(line)
                    
                    # Extract the required parameters
                    element_id = load_data.get("member_uid")
                    load_type = load_data.get("load_type")
                    
                    if element_id not in elements_dict:
                        print(f"Warning: Member load specified for non-existent element {element_id}")
                        line_idx += 1
                        continue
                        
                    element = elements_dict[element_id]
                    
                    # Get common parameters
                    load_case = load_data.get("case", "D")
                    direction = load_data.get("direction", "Y").upper()
                    location_percent = load_data.get("location_percent", False)
                    
                    print(f"Processing JSON member load for element {element_id}, type {load_type}")
                    
                    if load_type == 3:  # Distributed load
                        # Parse the load intensity with units
                        w1_with_units = parse_value_with_units(load_data.get("wi", "0"))
                        w1 = w1_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units, pint.Quantity) else w1_with_units
                        
                        # Check if wj exists, otherwise use wi for uniform load
                        if "wj" in load_data:
                            w2_with_units = parse_value_with_units(load_data.get("wj"))
                            w2 = w2_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units, pint.Quantity) else w2_with_units
                        else:
                            w2 = w1  # Uniform load
                        
                        # Parse start and end positions
                        a = load_data.get("a", 0.0)
                        b = load_data.get("b", element.length)
                        
                        # If positions are in percentages, convert to actual distances
                        if location_percent:
                            print(f"  Converting positions from percentages: a={a}%, b={b}%")
                            a = a / 100.0 * element.length
                            b = b / 100.0 * element.length
                        
                        print(f"  Element {element_id}: Distributed load w1={w1}, w2={w2}, a={a}{LENGTH_UNIT}, b={b}{LENGTH_UNIT}")
                        
                        # Apply load in appropriate direction
                        if direction.upper() == "X":
                            element.add_distributed_load(w1, w2, a, b, load_case, direction="xx")
                        else:
                            element.add_distributed_load(w1, w2, a, b, load_case)
                        
                    elif load_type == 1:  # Point load
                        # Parse force magnitude
                        p_with_units = parse_value_with_units(load_data.get("p", "0"))
                        p = p_with_units.to(FORCE_UNIT).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units
                        
                        # Parse position
                        a = load_data.get("a", 0.0)
                        
                        # Convert percentage to actual position if needed
                        if location_percent:
                            a = a / 100.0 * element.length
                        
                        print(f"  Element {element_id}: Point load p={p}{FORCE_UNIT}, position={a}{LENGTH_UNIT}")
                        
                        # Apply in appropriate direction
                        if direction.upper() == "X":
                            element.add_point_load(p, a, load_case, direction="xx")
                        else:
                            element.add_point_load(p, a, load_case)
                    
                    elif load_type == 2:  # Point moment
                        # Parse moment magnitude
                        m_with_units = parse_value_with_units(load_data.get("m", "0"))
                        m = m_with_units.to(MOMENT_UNIT).magnitude if isinstance(m_with_units, pint.Quantity) else m_with_units
                        
                        # Parse position
                        a = load_data.get("a", 0.0)
                        
                        # Convert percentage to actual position if needed
                        if location_percent:
                            a = a / 100.0 * element.length
                        
                        print(f"  Element {element_id}: Point moment m={m}{MOMENT_UNIT}, position={a}{LENGTH_UNIT}")
                        element.add_point_moment(m, a, load_case)
                    
                    else:
                        print(f"  Warning: Unsupported load type {load_type} in JSON format")
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON member load: {e}")
                except Exception as e:
                    print(f"Error processing JSON member load: {e}")
            else:
                # Original comma-separated format
                parts = [x.strip() for x in line.split(',')]
                
                # Keep the existing code for comma-separated format...
                # [Original code omitted for brevity]
                
                # Fallback to existing code for non-JSON format
                try:
                    element_id = int(parts[0])
                    
                    if element_id not in elements_dict:
                        print(f"Warning: Member load specified for non-existent element {element_id}")
                        line_idx += 1
                        continue
                        
                    element = elements_dict[element_id]
                    load_type = int(parts[1])
                    
                    # Map load type to appropriate load class based on the corrected mapping
                    # [Original code for handling different load types]
                    # Keep your existing if-elif blocks for each load type
                    # This part remains unchanged
                    
                except Exception as e:
                    print(f"  Error applying load to element: {e}")
            
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
        print(f"Node {node.uid:2d}  |  {rx_status:5s} |  {ry_status:5s} |  {rz_status:5s}")
     
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
