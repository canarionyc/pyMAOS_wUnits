def export_results_to_json(structure, load_cases=None, output_file=None):
    """
    Export calculation results to a JSON file
    
    Parameters
    ----------
    structure : R2Structure
        The analyzed structure containing results
    load_cases : list, optional
        List of load case names to include in export (defaults to all)
    output_file : str, optional
        Path to output file (defaults to 'results.json')
        
    Returns
    -------
    str
        Path to the created JSON file
    """
    import json
    import numpy as np
    
    if output_file is None:
        output_file = 'results.json'
    
    # Prepare data structure for JSON
    results = {
        "units": structure.units,
        "nodes": {},
        "members": {},
        "analysis_info": {
            "dof": structure.NDOF,
            "num_nodes": structure.NJ,
            "num_members": structure.NM
        }
    }
    
    # Helper function to safely extract 6 force values from various array formats
    def extract_force_values(force_array):
        """Extract 6 force values from various array formats"""
        values = [0.0] * 6
        
        try:
            if isinstance(force_array, np.matrix):
                # Handle numpy matrix - convert to array first
                array_data = np.array(force_array)
                flat_data = array_data.flatten()
                for i in range(min(6, len(flat_data))):
                    values[i] = float(flat_data[i])
            elif isinstance(force_array, np.ndarray):
                # Handle numpy array
                flat_data = force_array.flatten()
                for i in range(min(6, len(flat_data))):
                    values[i] = float(flat_data[i])
            elif hasattr(force_array, '__iter__'):
                # It's some other iterable
                for i, val in enumerate(force_array):
                    if i >= 6: break
                    values[i] = float(val)
            else:
                # Single value or unknown type
                values[0] = float(force_array)
                
            return values
        except Exception as e:
            print(f"Error extracting force values: {e}")
            return values
    
    # Get all load combination names if not specified
    if load_cases is None:
        load_cases = set()
        for node in structure.nodes:
            if hasattr(node, 'displacements'):
                load_cases.update(node.displacements.keys())
        load_cases = list(load_cases)
    # Handle case where LoadCombo objects are passed instead of names
    elif all(hasattr(lc, 'name') for lc in load_cases):
        load_cases = [lc.name for lc in load_cases]
    
    # Extract node results
    for node in structure.nodes:
        node_data = {
            "coordinates": {
                "x": float(node.x),
                "y": float(node.y)
            },
            "restraints": [int(r) for r in node.restraints],
            "displacements": {},
            "reactions": {}
        }
        
        # Add displacements for each load combo
        if hasattr(node, 'displacements'):
            for lc in load_cases:
                if lc in node.displacements:
                    node_data["displacements"][lc] = {
                        "ux": float(node.displacements[lc][0]),
                        "uy": float(node.displacements[lc][1]),
                        "rz": float(node.displacements[lc][2])
                    }
        
        # Add reactions for each load combo
        if hasattr(node, 'reactions'):
            for lc in load_cases:
                if lc in node.reactions:
                    node_data["reactions"][lc] = {
                        "rx": float(node.reactions[lc][0]),
                        "ry": float(node.reactions[lc][1]),
                        "mz": float(node.reactions[lc][2])
                    }
        
        results["nodes"][str(node.uid)] = node_data
    
    # Extract member results
    for member in structure.members:
        member_data = {
            "connectivity": {
                "i_node": int(member.inode.uid),
                "j_node": int(member.jnode.uid)
            },
            "properties": {
                "length": float(member.length),
                "type": member.type if hasattr(member, 'type') else "FRAME"
            },
            "forces": {}
        }
        
        # Add material and section properties
        if hasattr(member, 'material'):
            member_data["properties"]["material"] = {
                "id": str(member.material.uid),
                "E": float(member.material.E)
            }
        
        if hasattr(member, 'section'):
            member_data["properties"]["section"] = {
                "id": str(member.section.uid),
                "area": float(member.section.Area),
                "Ixx": float(member.section.Ixx)
            }
        
        # Add member hinges if applicable
        if hasattr(member, 'hinges'):
            member_data["properties"]["hinges"] = [int(h) for h in member.hinges]
        
        # Add forces for each load combo
        for lc in load_cases:
            # Use already calculated values from end_forces_global
            if hasattr(member, 'end_forces_global') and lc in member.end_forces_global:
                try:
                    # Get pre-calculated global forces
                    forces_global = member.end_forces_global[lc]
                    global_values = extract_force_values(forces_global)
                    
                    member_data["forces"][lc] = {
                        "global": {
                            "i_node": {
                                "fx": global_values[0],
                                "fy": global_values[1],
                                "mz": global_values[2]
                            },
                            "j_node": {
                                "fx": global_values[3],
                                "fy": global_values[4],
                                "mz": global_values[5]
                            }
                        }
                    }
                    
                    # Add local forces if available
                    if hasattr(member, 'end_forces_local') and lc in member.end_forces_local:
                        forces_local = member.end_forces_local[lc]
                        local_values = extract_force_values(forces_local)
                        
                        member_data["forces"][lc]["local"] = {
                            "i_node": {
                                "fx": local_values[0],
                                "fy": local_values[1],
                                "mz": local_values[2]
                            },
                            "j_node": {
                                "fx": local_values[3],
                                "fy": local_values[4],
                                "mz": local_values[5]
                            }
                        }
                    
                    # Add distributed results if available
                    try:
                        # Only proceed if the member has generated these functions
                        if (hasattr(member, 'A') and lc in member.A and
                            hasattr(member, 'Vy') and lc in member.Vy and
                            hasattr(member, 'Mz') and lc in member.Mz):
                            
                            ax_func = member.A[lc]
                            vy_func = member.Vy[lc]
                            mz_func = member.Mz[lc]
                            
                            if ax_func and vy_func and mz_func:
                                num_points = 21  # Number of points to sample along the member
                                x_vals = [member.length * i / (num_points - 1) for i in range(num_points)]
                                
                                member_data["forces"][lc]["distributed"] = {
                                    "positions": x_vals,
                                    "axial": [float(ax_func.evaluate(x)) for x in x_vals],
                                    "shear": [float(vy_func.evaluate(x)) for x in x_vals],
                                    "moment": [float(mz_func.evaluate(x)) for x in x_vals]
                                }
                                
                                # Add extreme moment values if available
                                if hasattr(member, 'Mzextremes'):
                                    try:
                                        extremes = member.Mzextremes(lc)
                                        if extremes:
                                            member_data["forces"][lc]["extremes"] = {
                                                "max_moment": {
                                                    "position": float(extremes["MaxM"][0]),
                                                    "value": float(extremes["MaxM"][1])
                                                },
                                                "min_moment": {
                                                    "position": float(extremes["MinM"][0]),
                                                    "value": float(extremes["MinM"][1])
                                                }
                                            }
                                    except Exception as e:
                                        print(f"Error calculating moment extremes for member {member.uid}: {e}")
                    except Exception as e:
                        print(f"Could not generate distributed results for member {member.uid}: {e}")
                            
                except Exception as e:
                    print(f"Error extracting forces for member {member.uid}: {e}")
        
        results["members"][str(member.uid)] = member_data
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results successfully exported to {output_file}")
    return output_file