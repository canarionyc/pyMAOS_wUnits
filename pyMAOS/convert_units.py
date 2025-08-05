# #!/usr/bin/env python3
# """
# Convert JSON structural model from any unit system to SI units.
# """
# import os
# import json
# import re
# import sys
# import argparse
# from pathlib import Path
# import pint
#
# # Initialize unit registry
# from pyMAOS import unit_manager
#
# Q_ = unit_manager.ureg.Quantity
#
# # Define dimension mappings - what fields have what physical dimensions
# DIMENSION_REGISTRY = {
#     "nodes": {
#         "x": "length",
#         "y": "length"
#     },
#     "materials": {
#         "E": "pressure"
#     },
#     "sections": {
#         "area": "area",
#         "ixx": "moment_of_inertia"
#     },
#     "joint_loads": {
#         "fx": "force",
#         "fy": "force",
#         "mz": "moment"
#     },
#     "member_loads": {
#         "wi": "distributed_load",
#         "wj": "distributed_load",
#         "a": "length",
#         "b": "length"
#     }
# }
#
# # Define SI unit for each dimension
# SI_UNITS = {
#     "length": "m",
#     "area": "m^2",
#     "moment_of_inertia": "m^4",
#     "force": "N",
#     "pressure": "Pa",
#     "moment": "N*m",
#     "distributed_load": "N/m",
#     "angle": "rad",
#     "rotation": "rad"
# }
#
# def parse_value_with_units(value_string):
#     """Parse a string that may contain a value with units."""
#     if not isinstance(value_string, str):
#         return value_string
#
#     # Match pattern: [numeric value][units]
#     match = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(.*)', value_string.strip())
#
#     if match:
#         value_str, unit_str = match.groups()
#         value = float(value_str)
#
#         if unit_str and unit_str.strip():
#             try:
#                 # Create quantity with units
#                 return Q_(value, unit_str)
#             except:
#                 print(f"Warning: Could not parse unit '{unit_str}', treating as dimensionless")
#                 return value
#         return value
#
#     # If no match, try to evaluate as a simple numeric expression
#     try:
#         return float(eval(value_string))
#     except:
#         raise ValueError(f"Could not parse value: {value_string}")
#
# def convert_to_si(value, dimension, unit_system, default_units):
#     """Convert a value to SI units based on its dimension."""
#     if isinstance(value, pint.Quantity):
#         # Value already has explicit units
#         si_unit = SI_UNITS.get(dimension)
#         if si_unit:
#             return value.to(si_unit).magnitude
#     elif isinstance(value, (int, float)):
#         # Value has implicit units based on default_units
#         if dimension in default_units:
#             implicit_unit = default_units[dimension]
#             quantity = Q_(value, implicit_unit)
#             si_unit = SI_UNITS.get(dimension)
#             if si_unit:
#                 return quantity.to(si_unit).magnitude
#
#     # No conversion needed or possible
#     return value
#
# def convert_json_to_si(json_data, output_file=None):
#     """Convert all values in a JSON structure to SI units."""
#     # Extract unit system and default units
#     unit_system = json_data.get("unit_system", "SI").lower()
#     default_units = json_data.get("units", {})
#
#     # Set up output data structure
#     si_data = json_data.copy()
#
#     # Replace unit system
#     si_data["unit_system"] = "SI"
#
#     # Update units section to SI
#     si_data["units"] = {
#         "length": "m",
#         "pressure": "Pa",
#         "distance": "m",
#         "force": "N"
#     }
#
#     # Process each section with physical dimensions
#     for section_name, field_dimensions in DIMENSION_REGISTRY.items():
#         if section_name in json_data:
#             section_data = json_data[section_name]
#
#             # Handle array sections
#             if isinstance(section_data, list):
#                 for i, item in enumerate(section_data):
#                     for field, dimension in field_dimensions.items():
#                         if field in item:
#                             original_value = item[field]
#                             print(f"Converting {section_name}[{i}].{field} from {original_value} to SI units")
#                             # Parse value if it's a string with units
#                             if isinstance(original_value, str):
#                                 parsed_value = parse_value_with_units(original_value)
#                                 if isinstance(parsed_value, pint.Quantity):
#                                     # Convert to SI
#                                     si_value = parsed_value.to(SI_UNITS[dimension]).magnitude
#                                     print(f"Converted {section_name}[{i}].{field} to SI: {si_value} {SI_UNITS[dimension]}")
#                                     si_data[section_name][i][field] = si_value
#                                     continue
#
#                             # Handle numeric values with implicit units
#                             print(f"Converting {section_name}[{i}].{field} original_value {original_value} dimension {dimension} with implicit units")
#                             si_data[section_name][i][field] = convert_to_si(
#                                 original_value, dimension, unit_system, default_units)
#
#     # Write to output file or print to stdout
#     if output_file:
#         with open(output_file, 'w') as f:
#             json.dump(si_data, f, indent=2)
#         print(f"Converted data written to {output_file}")
#     else:
#         print(json.dumps(si_data, indent=2))
#     print(json.dumps(si_data, indent=2))
#     return si_data
#
# def convert_si_to_display_units(si_data, output_file=None, display_units=None):
#     """
#     Convert structural analysis results from SI units to display units
#
#     Parameters
#     ----------
#     si_data : dict or str
#         JSON data structure or path to JSON file with SI units
#     output_file : str, optional
#         Path to output file (defaults to None, which returns converted data)
#     display_units : dict, optional
#         Dictionary mapping dimensions to display units (defaults to using
#         current display units from pyMAOS.units_mod)
#
#     Returns
#     -------
#     dict
#         Data structure with values converted to display units
#     """
#     import json
#     from copy import deepcopy
#     import pint
#
#     from pyMAOS.units_mod import ureg
#
#     # Handle case where si_data is a file path
#     if isinstance(si_data, str):
#         with open(si_data, 'r') as f:
#             si_data = json.load(f)
#
#     # Get display units if not provided
#     if display_units is None:
#         try:
#             from pyMAOS.units_mod import DISPLAY_UNITS
#             display_units = DISPLAY_UNITS
#         except ImportError:
#             # Fallback to some standard display units
#             display_units = {
#                 "force": "lbf",
#                 "length": "ft",
#                 "moment": "lbf*ft",
#                 "pressure": "psi",
#                 "area": "in^2",
#                 "moment_of_inertia": "in^4",
#                 "distributed_load": "lbf/ft",
#                 "rotation": "rad"
#             }
#
#     # Define SI units for each dimension
#     si_units = {
#         "force": "N",
#         "length": "m",
#         "moment": "N*m",
#         "pressure": "Pa",
#         "area": "m^2",
#         "moment_of_inertia": "m^4",
#         "distributed_load": "N/m",
#         "rotation": "rad"
#     }
#
#     # Create a copy of the data to modify
#     display_data = deepcopy(si_data)
#
#     # Update the units section
#     if "units" in display_data:
#         display_data["units"] = display_units
#
#     # Define mapping of JSON paths to dimensions
#     conversion_map = {
#         # Node data
#         "nodes.coordinates.x": "length",
#         "nodes.coordinates.y": "length",
#         "nodes.displacements.ux": "length",
#         "nodes.displacements.uy": "length",
#         "nodes.displacements.rz": "rotation",
#         "nodes.reactions.rx": "force",
#         "nodes.reactions.ry": "force",
#         "nodes.reactions.mz": "moment",
#
#         # Member properties
#         "members.properties.length": "length",
#         "members.properties.material.E": "pressure",
#         "members.properties.section.area": "area",
#         "members.properties.section.Ixx": "moment_of_inertia",
#
#         # Member forces
#         "members.forces.global.i_node.fx": "force",
#         "members.forces.global.i_node.fy": "force",
#         "members.forces.global.i_node.mz": "moment",
#         "members.forces.global.j_node.fx": "force",
#         "members.forces.global.j_node.fy": "force",
#         "members.forces.global.j_node.mz": "moment",
#         "members.forces.local.i_node.fx": "force",
#         "members.forces.local.i_node.fy": "force",
#         "members.forces.local.i_node.mz": "moment",
#         "members.forces.local.j_node.fx": "force",
#         "members.forces.local.j_node.fy": "force",
#         "members.forces.local.j_node.mz": "moment",
#
#         # Distributed results
#         "members.forces.distributed.positions": "length",
#         "members.forces.distributed.axial": "force",
#         "members.forces.distributed.shear": "force",
#         "members.forces.distributed.moment": "moment",
#
#         # Extremes
#         "members.forces.extremes.max_moment.position": "length",
#         "members.forces.extremes.max_moment.value": "moment",
#         "members.forces.extremes.min_moment.position": "length",
#         "members.forces.extremes.min_moment.value": "moment"
#     }
#
#     # Convert a single value using pint
#     def convert_value(value, from_unit, to_unit):
#         try:
#             # Skip conversion for values very close to zero
#             if abs(float(value)) < 1e-12:
#                 return 0.0
#
#             # Convert using pint
#             quantity = ureg.Quantity(float(value), from_unit)
#             return quantity.to(to_unit).magnitude
#         except Exception as e:
#             print(f"Error converting {value} from {from_unit} to {to_unit}: {e}")
#             return value
#
#     # Process nodes
#     for node_id, node_data in display_data.get("nodes", {}).items():
#         # Process coordinates
#         if "coordinates" in node_data:
#             node_data["coordinates"]["x"] = convert_value(
#                 node_data["coordinates"]["x"],
#                 si_units["length"],
#                 display_units.get("length", si_units["length"])
#             )
#             node_data["coordinates"]["y"] = convert_value(
#                 node_data["coordinates"]["y"],
#                 si_units["length"],
#                 display_units.get("length", si_units["length"])
#             )
#
#         # Process displacements for each load case
#         for lc, displ in node_data.get("displacements", {}).items():
#             displ["ux"] = convert_value(
#                 displ["ux"],
#                 si_units["length"],
#                 display_units.get("length", si_units["length"])
#             )
#             displ["uy"] = convert_value(
#                 displ["uy"],
#                 si_units["length"],
#                 display_units.get("length", si_units["length"])
#             )
#             displ["rz"] = convert_value(
#                 displ["rz"],
#                 si_units["rotation"],
#                 display_units.get("rotation", si_units["rotation"])
#             )
#
#         # Process reactions for each load case
#         for lc, react in node_data.get("reactions", {}).items():
#             react["rx"] = convert_value(
#                 react["rx"],
#                 si_units["force"],
#                 display_units.get("force", si_units["force"])
#             )
#             react["ry"] = convert_value(
#                 react["ry"],
#                 si_units["force"],
#                 display_units.get("force", si_units["force"])
#             )
#             react["mz"] = convert_value(
#                 react["mz"],
#                 si_units["moment"],
#                 display_units.get("moment", si_units["moment"])
#             )
#
#     # Process members
#     for member_id, member_data in display_data.get("members", {}).items():
#         # Process properties
#         props = member_data.get("properties", {})
#         if "length" in props:
#             props["length"] = convert_value(
#                 props["length"],
#                 si_units["length"],
#                 display_units.get("length", si_units["length"])
#             )
#
#         # Process material properties
#         material = props.get("material", {})
#         if "E" in material:
#             material["E"] = convert_value(
#                 material["E"],
#                 si_units["pressure"],
#                 display_units.get("pressure", si_units["pressure"])
#             )
#
#         # Process section properties
#         section = props.get("section", {})
#         if "area" in section:
#             section["area"] = convert_value(
#                 section["area"],
#                 si_units["area"],
#                 display_units.get("area", si_units["area"])
#             )
#         if "Ixx" in section:
#             section["Ixx"] = convert_value(
#                 section["Ixx"],
#                 si_units["moment_of_inertia"],
#                 display_units.get("moment_of_inertia", si_units["moment_of_inertia"])
#             )
#
#         # Process forces for each load case
#         for lc, forces in member_data.get("forces", {}).items():
#             # Process global forces
#             if "global" in forces:
#                 # i-node forces
#                 i_node = forces["global"].get("i_node", {})
#                 if "fx" in i_node:
#                     i_node["fx"] = convert_value(
#                         i_node["fx"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "fy" in i_node:
#                     i_node["fy"] = convert_value(
#                         i_node["fy"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "mz" in i_node:
#                     i_node["mz"] = convert_value(
#                         i_node["mz"],
#                         si_units["moment"],
#                         display_units.get("moment", si_units["moment"])
#                     )
#
#                 # j-node forces
#                 j_node = forces["global"].get("j_node", {})
#                 if "fx" in j_node:
#                     j_node["fx"] = convert_value(
#                         j_node["fx"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "fy" in j_node:
#                     j_node["fy"] = convert_value(
#                         j_node["fy"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "mz" in j_node:
#                     j_node["mz"] = convert_value(
#                         j_node["mz"],
#                         si_units["moment"],
#                         display_units.get("moment", si_units["moment"])
#                     )
#
#             # Process local forces
#             if "local" in forces:
#                 # i-node forces
#                 i_node = forces["local"].get("i_node", {})
#                 if "fx" in i_node:
#                     i_node["fx"] = convert_value(
#                         i_node["fx"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "fy" in i_node:
#                     i_node["fy"] = convert_value(
#                         i_node["fy"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "mz" in i_node:
#                     i_node["mz"] = convert_value(
#                         i_node["mz"],
#                         si_units["moment"],
#                         display_units.get("moment", si_units["moment"])
#                     )
#
#                 # j-node forces
#                 j_node = forces["local"].get("j_node", {})
#                 if "fx" in j_node:
#                     j_node["fx"] = convert_value(
#                         j_node["fx"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "fy" in j_node:
#                     j_node["fy"] = convert_value(
#                         j_node["fy"],
#                         si_units["force"],
#                         display_units.get("force", si_units["force"])
#                     )
#                 if "mz" in j_node:
#                     j_node["mz"] = convert_value(
#                         j_node["mz"],
#                         si_units["moment"],
#                         display_units.get("moment", si_units["moment"])
#                     )
#
#             # Process distributed results
#             if "distributed" in forces:
#                 dist = forces["distributed"]
#
#                 # Convert positions array
#                 if "positions" in dist:
#                     dist["positions"] = [
#                         convert_value(
#                             pos,
#                             si_units["length"],
#                             display_units.get("length", si_units["length"])
#                         )
#                         for pos in dist["positions"]
#                     ]
#
#                 # Convert axial force array
#                 if "axial" in dist:
#                     dist["axial"] = [
#                         convert_value(
#                             ax,
#                             si_units["force"],
#                             display_units.get("force", si_units["force"])
#                         )
#                         for ax in dist["axial"]
#                     ]
#
#                 # Convert shear force array
#                 if "shear" in dist:
#                     dist["shear"] = [
#                         convert_value(
#                             sh,
#                             si_units["force"],
#                             display_units.get("force", si_units["force"])
#                         )
#                         for sh in dist["shear"]
#                     ]
#
#                 # Convert moment array
#                 if "moment" in dist:
#                     dist["moment"] = [
#                         convert_value(
#                             mom,
#                             si_units["moment"],
#                             display_units.get("moment", si_units["moment"])
#                         )
#                         for mom in dist["moment"]
#                     ]
#
#             # Process extreme values
#             if "extremes" in forces:
#                 extremes = forces["extremes"]
#
#                 # Max moment
#                 if "max_moment" in extremes:
#                     max_m = extremes["max_moment"]
#                     if "position" in max_m:
#                         max_m["position"] = convert_value(
#                             max_m["position"],
#                             si_units["length"],
#                             display_units.get("length", si_units["length"])
#                         )
#                     if "value" in max_m:
#                         max_m["value"] = convert_value(
#                             max_m["value"],
#                             si_units["moment"],
#                             display_units.get("moment", si_units["moment"])
#                         )
#
#                 # Min moment
#                 if "min_moment" in extremes:
#                     min_m = extremes["min_moment"]
#                     if "position" in min_m:
#                         min_m["position"] = convert_value(
#                             min_m["position"],
#                             si_units["length"],
#                             display_units.get("length", si_units["length"])
#                         )
#                     if "value" in min_m:
#                         min_m["value"] = convert_value(
#                             min_m["value"],
#                             si_units["moment"],
#                             display_units.get("moment", si_units["moment"])
#                         )
#
#     # Write to output file if provided
#     if output_file:
#         with open(output_file, 'w') as f:
#             json.dump(display_data, f, indent=2)
#         print(f"Converted results written to {output_file}")
#
#     return display_data
#
# if __name__ == "__main__":
#     status=0
#     """Command line interface for unit conversion."""
#     parser = argparse.ArgumentParser(description="Convert JSON structural model to SI units")
#     parser.add_argument("input_file", help="Input JSON file path")
#     parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
#     args = parser.parse_args()
#
#     # Determine output filename
#     if not args.output:
#         input_path = Path(args.input_file)
#         args.output = input_path.with_stem(f"{input_path.stem}_SI").with_suffix('.json')
#
#     if args.output == args.input_file:
#         print(f"Warning: Output file {args.output} is the same as input file {args.input_file}.")
#
#     if args.output.exists():
#         print(f"Warning: Output file {args.output} already exists. It will be overwritten.")
#         args.output.unlink()
#     try:
#         with open(args.input_file, 'r') as f:
#             json_data = json.load(f)
#
#         convert_json_to_si(json_data, args.output)
#         print(f"Successfully converted {args.input_file} to SI units")
#
#     except FileNotFoundError:
#         print(f"Error: File {args.input_file} not found")
#         status = 1
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON in {args.input_file}")
#         status = 1
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         status = 1
#
#     print(args.output.read_text())
#
#     exit(status)
