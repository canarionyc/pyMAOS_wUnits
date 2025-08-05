def export_results_to_json(structure, load_cases=None, output_file=None, convert_to_display_units=False, display_units=None):
    """
    Export calculation results to a JSON file with Pint unit support,
    including units next to magnitude values

    Parameters
    ----------
    structure : R2Structure
        The analyzed structure containing results
    load_cases : list, optional
        List of load case names to include in export (defaults to all)
    output_file : str, optional
        Path to output file (defaults to 'results.json')
    convert_to_display_units : bool, optional
        Whether to convert values to display units (default: False)
    display_units : dict, optional
        Dictionary of display units to use (default: None, uses system defaults)

    Returns
    -------
    str
        Path to the created JSON file
    """
    import json
    import numpy as np
    import os

    # Import the centralized unit manager
    from pyMAOS import unit_manager
    ureg = unit_manager.ureg

    if output_file is None:
        output_file = 'results.json'

    print(f"Starting export to JSON file: {output_file}")

    # Get default display units if needed but not provided
    if convert_to_display_units and display_units is None:
        try:
            from pyMAOS import unit_manager
            display_units = unit_manager.current_system
        except ImportError:
            # Fallback to some standard display units
            display_units = {
                "force": "kN",
                "length": "m",
                "moment": "kN*m",
                "pressure": "MPa",
                "area": "cm^2",
                "moment_of_inertia": "cm^4",
                "distributed_load": "kN/m",
                "rotation": "rad"
            }
            print(f"Using default display units: {display_units}")

    # Define internal units based on the unit manager
    internal_units = {
        "force": str(unit_manager.INTERNAL_FORCE_UNIT),
        "length": str(unit_manager.INTERNAL_LENGTH_UNIT),
        "moment": str(unit_manager.INTERNAL_MOMENT_UNIT),
        "pressure": str(unit_manager.INTERNAL_PRESSURE_UNIT),
        "area": f"{unit_manager.INTERNAL_LENGTH_UNIT}^2",
        "moment_of_inertia": str(unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT),
        "distributed_load": str(unit_manager.INTERNAL_DISTRIBUTED_LOAD_UNIT),
        "rotation": str(unit_manager.INTERNAL_ROTATION_UNIT)
    }

    print(f"Internal units: {internal_units}")
    print(f"Display units (if converting): {display_units}")

    # Prepare data structure for JSON
    results = {
        "units": display_units if convert_to_display_units else internal_units,
        "nodes": {},
        "members": {},
        "analysis_info": {
            "dof": structure.NDOF,
            "num_nodes": structure.NJ,
            "num_members": structure.NM
        }
    }

    # Helper function to safely extract value from Pint quantities or regular values
    def extract_value(val):
        """Extract numerical value from Pint quantities or regular values"""
        try:
            # Check if value has a 'magnitude' attribute (Pint Quantity)
            if hasattr(val, 'magnitude'):
                return float(val.magnitude)
            # For numpy types
            elif isinstance(val, (np.number, np.ndarray, np.matrix)):
                return float(val)
            else:
                return float(val)
        except Exception as e:
            print(f"Error extracting value: {e}, using 0.0")
            return 0.0

    # Unit conversion helper function
    def convert_unit_with_info(value, dimension):
        """Convert value between unit systems and return both value and unit"""
        try:
            # Determine the unit based on conversion settings
            unit = display_units[dimension] if convert_to_display_units else internal_units[dimension]

            # Check if value is close to zero without direct float conversion
            # Handle Pint quantities properly
            if hasattr(value, 'magnitude'):
                magnitude = value.magnitude
            else:
                try:
                    magnitude = float(value)
                except (TypeError, ValueError):
                    print(f"Warning: Could not convert {value} to float, using 0")
                    magnitude = 0.0

            # Skip conversion for values very close to zero
            if abs(magnitude) < 1e-12:
                return {"value": 0.0, "unit": unit}

            # Extract the raw value
            raw_value = extract_value(value)

            # If conversion is requested, perform it
            if convert_to_display_units:
                source_unit = internal_units[dimension]
                target_unit = display_units[dimension]

                # Create quantity with source unit and convert to target unit
                quantity = ureg.Quantity(raw_value, source_unit)
                raw_value = quantity.to(target_unit).magnitude

            # Return both value and unit
            return {"value": raw_value, "unit": unit}
        except Exception as e:
            print(f"Error converting {value} from {internal_units.get(dimension)} to {display_units.get(dimension)}: {e}")
            unit = internal_units.get(dimension)
            return {"value": extract_value(value), "unit": unit}

    # Helper function to safely extract 6 force values from various array formats
    def extract_force_values(force_array):
        """Extract 6 force values from various array formats with Pint support"""
        values = [0.0] * 6

        try:
            # Handle Pint quantities in arrays
            if hasattr(force_array, 'magnitude'):
                force_array = force_array.magnitude

            if isinstance(force_array, np.matrix):
                array_data = np.array(force_array)
                flat_data = array_data.flatten()
                for i in range(min(6, len(flat_data))):
                    values[i] = extract_value(flat_data[i])
            elif isinstance(force_array, np.ndarray):
                flat_data = force_array.flatten()
                for i in range(min(6, len(flat_data))):
                    values[i] = extract_value(flat_data[i])
            elif hasattr(force_array, '__iter__'):
                for i, val in enumerate(force_array):
                    if i >= 6: break
                    values[i] = extract_value(val)
            else:
                # Single value or unknown type
                values[0] = extract_value(force_array)

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
        print(f"Found {len(load_cases)} load cases: {load_cases}")
    # Handle case where LoadCombo objects are passed instead of names
    elif all(hasattr(lc, 'name') for lc in load_cases):
        load_cases = [lc.name for lc in load_cases]

    # Extract node results
    print(f"Processing {len(structure.nodes)} nodes")
    for node in structure.nodes:
        node_data = {
            "coordinates": {
                "x": convert_unit_with_info(node.x, "length"),
                "y": convert_unit_with_info(node.y, "length")
            },
            "restraints": [int(r) for r in node.restraints] if hasattr(node, 'restraints') else [0, 0, 0],
            "displacements": {},
            "reactions": {}
        }

        # Add displacements for each load combo
        if hasattr(node, 'displacements'):
            for lc in load_cases:
                if lc in node.displacements:
                    node_data["displacements"][lc] = {
                        "ux": convert_unit_with_info(node.displacements[lc][0], "length"),
                        "uy": convert_unit_with_info(node.displacements[lc][1], "length"),
                        "rz": convert_unit_with_info(node.displacements[lc][2], "rotation")
                    }

        # Add reactions for each load combo
        if hasattr(node, 'reactions'):
            for lc in load_cases:
                if lc in node.reactions:
                    node_data["reactions"][lc] = {
                        "rx": convert_unit_with_info(node.reactions[lc][0], "force"),
                        "ry": convert_unit_with_info(node.reactions[lc][1], "force"),
                        "mz": convert_unit_with_info(node.reactions[lc][2], "moment")
                    }

        results["nodes"][str(node.uid)] = node_data

    # Extract member results
    print(f"Processing {len(structure.members)} members")
    for member in structure.members:
        member_data = {
            "connectivity": {
                "i_node": int(member.inode.uid),
                "j_node": int(member.jnode.uid)
            },
            "properties": {
                "length": convert_unit_with_info(member.length, "length"),
                "type": member.type if hasattr(member, 'type') else "FRAME"
            },
            "forces": {}
        }

        # Add material and section properties
        if hasattr(member, 'material'):
            member_data["properties"]["material"] = {
                "id": str(member.material.uid),
                "E": convert_unit_with_info(member.material.E, "pressure")
            }

        if hasattr(member, 'section'):
            member_data["properties"]["section"] = {
                "id": str(member.section.uid),
                "area": convert_unit_with_info(member.section.Area, "area"),
                "Ixx": convert_unit_with_info(member.section.Ixx, "moment_of_inertia")
            }

        # Add member hinges if applicable
        if hasattr(member, 'hinges'):
            member_data["properties"]["hinges"] = [int(h) for h in member.hinges]

        # Add forces for each load combo
        for lc in load_cases:
            # Process member forces if available
            if hasattr(member, 'member_forces') and lc in member.member_forces:
                forces = member.member_forces[lc]
                force_values = extract_force_values(forces)

                member_data["forces"][lc] = {
                    "local": {
                        "i_node": {
                            "fx": convert_unit_with_info(force_values[0], "force"),
                            "fy": convert_unit_with_info(force_values[1], "force"),
                            "mz": convert_unit_with_info(force_values[2], "moment")
                        },
                        "j_node": {
                            "fx": convert_unit_with_info(force_values[3], "force"),
                            "fy": convert_unit_with_info(force_values[4], "force"),
                            "mz": convert_unit_with_info(force_values[5], "moment")
                        }
                    }
                }

            # Add global forces if available
            if hasattr(member, 'end_forces_global') and lc in member.end_forces_global:
                forces_global = member.end_forces_global[lc]
                global_values = extract_force_values(forces_global)

                if "forces" not in member_data:
                    member_data["forces"] = {}
                if lc not in member_data["forces"]:
                    member_data["forces"][lc] = {}

                member_data["forces"][lc]["global"] = {
                    "i_node": {
                        "fx": convert_unit_with_info(global_values[0], "force"),
                        "fy": convert_unit_with_info(global_values[1], "force"),
                        "mz": convert_unit_with_info(global_values[2], "moment")
                    },
                    "j_node": {
                        "fx": convert_unit_with_info(global_values[3], "force"),
                        "fy": convert_unit_with_info(global_values[4], "force"),
                        "mz": convert_unit_with_info(global_values[5], "moment")
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
                        length = extract_value(member.length)
                        x_vals = [length * i / (num_points - 1) for i in range(num_points)]

                        if lc not in member_data["forces"]:
                            member_data["forces"][lc] = {}

                        member_data["forces"][lc]["distributed"] = {
                            "positions": [convert_unit_with_info(x, "length") for x in x_vals],
                            "axial": [convert_unit_with_info(ax_func.evaluate(x), "force") for x in x_vals],
                            "shear": [convert_unit_with_info(vy_func.evaluate(x), "force") for x in x_vals],
                            "moment": [convert_unit_with_info(mz_func.evaluate(x), "moment") for x in x_vals]
                        }

                        # Add extreme moment values if available
                        if hasattr(member, 'Mzextremes'):
                            try:
                                extremes = member.Mzextremes(lc)
                                if extremes:
                                    member_data["forces"][lc]["extremes"] = {
                                        "max_moment": {
                                            "position": convert_unit_with_info(extremes["MaxM"][0], "length"),
                                            "value": convert_unit_with_info(extremes["MaxM"][1], "moment")
                                        },
                                        "min_moment": {
                                            "position": convert_unit_with_info(extremes["MinM"][0], "length"),
                                            "value": convert_unit_with_info(extremes["MinM"][1], "moment")
                                        }
                                    }
                            except Exception as e:
                                print(f"Error calculating moment extremes for member {member.uid}: {e}")
            except Exception as e:
                print(f"Could not generate distributed results for member {member.uid}: {e}")

        results["members"][str(member.uid)] = member_data

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write to JSON file
    print(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully exported to {output_file}")
    return output_file