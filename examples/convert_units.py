#!/usr/bin/env python3
"""
Convert JSON structural model from any unit system to SI units.
"""
import os
import json
import re
import sys
import argparse
from pathlib import Path
import pint

# Initialize unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define dimension mappings - what fields have what physical dimensions
DIMENSION_REGISTRY = {
    "nodes": {
        "x": "length",
        "y": "length"
    },
    "materials": {
        "E": "pressure"
    },
    "sections": {
        "area": "area",
        "ixx": "moment_of_inertia"
    },
    "joint_loads": {
        "fx": "force",
        "fy": "force",
        "mz": "moment"
    },
    "member_loads": {
        "wi": "distributed_load",
        "wj": "distributed_load",
        "a": "length",
        "b": "length"
    }
}

# Define SI unit for each dimension
SI_UNITS = {
    "length": "m",
    "area": "m^2",
    "moment_of_inertia": "m^4",
    "force": "N",
    "pressure": "Pa",
    "moment": "N*m",
    "distributed_load": "N/m",
    "angle": "rad",
    "rotation": "rad"
}

def parse_value_with_units(value_string):
    """Parse a string that may contain a value with units."""
    if not isinstance(value_string, str):
        return value_string
    
    # Match pattern: [numeric value][units]
    match = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(.*)', value_string.strip())
    
    if match:
        value_str, unit_str = match.groups()
        value = float(value_str)
        
        if unit_str and unit_str.strip():
            try:
                # Create quantity with units
                return Q_(value, unit_str)
            except:
                print(f"Warning: Could not parse unit '{unit_str}', treating as dimensionless")
                return value
        return value
    
    # If no match, try to evaluate as a simple numeric expression
    try:
        return float(eval(value_string))
    except:
        raise ValueError(f"Could not parse value: {value_string}")

def convert_to_si(value, dimension, unit_system, default_units):
    """Convert a value to SI units based on its dimension."""
    if isinstance(value, pint.Quantity):
        # Value already has explicit units
        si_unit = SI_UNITS.get(dimension)
        if si_unit:
            return value.to(si_unit).magnitude
    elif isinstance(value, (int, float)):
        # Value has implicit units based on default_units
        if dimension in default_units:
            implicit_unit = default_units[dimension]
            quantity = Q_(value, implicit_unit)
            si_unit = SI_UNITS.get(dimension)
            if si_unit:
                return quantity.to(si_unit).magnitude
    
    # No conversion needed or possible
    return value

def convert_json_to_si(json_data, output_file=None):
    """Convert all values in a JSON structure to SI units."""
    # Extract unit system and default units
    unit_system = json_data.get("unit_system", "SI").lower()
    default_units = json_data.get("units", {})
    
    # Set up output data structure
    si_data = json_data.copy()
    
    # Replace unit system
    si_data["unit_system"] = "SI"
    
    # Update units section to SI
    si_data["units"] = {
        "length": "m",
        "pressure": "Pa",
        "distance": "m",
        "force": "N"
    }
    
    # Process each section with physical dimensions
    for section_name, field_dimensions in DIMENSION_REGISTRY.items():
        if section_name in json_data:
            section_data = json_data[section_name]
            
            # Handle array sections
            if isinstance(section_data, list):
                for i, item in enumerate(section_data):
                    for field, dimension in field_dimensions.items():
                        if field in item:
                            original_value = item[field]
                            print(f"Converting {section_name}[{i}].{field} from {original_value} to SI units")
                            # Parse value if it's a string with units
                            if isinstance(original_value, str):
                                parsed_value = parse_value_with_units(original_value)
                                if isinstance(parsed_value, pint.Quantity):
                                    # Convert to SI
                                    si_value = parsed_value.to(SI_UNITS[dimension]).magnitude
                                    print(f"Converted {section_name}[{i}].{field} to SI: {si_value} {SI_UNITS[dimension]}")
                                    si_data[section_name][i][field] = si_value
                                    continue
                            
                            # Handle numeric values with implicit units
                            print(f"Converting {section_name}[{i}].{field} original_value {original_value} dimension {dimension} with implicit units")
                            si_data[section_name][i][field] = convert_to_si(
                                original_value, dimension, unit_system, default_units)
    
    # Write to output file or print to stdout
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(si_data, f, indent=2)
        print(f"Converted data written to {output_file}")
    else:
        print(json.dumps(si_data, indent=2))
    print(json.dumps(si_data, indent=2))
    return si_data

if __name__ == "__main__":
    status=0
    """Command line interface for unit conversion."""
    parser = argparse.ArgumentParser(description="Convert JSON structural model to SI units")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    args = parser.parse_args()
    
    # Determine output filename
    if not args.output:
        input_path = Path(args.input_file)
        args.output = input_path.with_stem(f"{input_path.stem}_SI").with_suffix('.json')
    
    if args.output == args.input_file:
        print(f"Warning: Output file {args.output} is the same as input file {args.input_file}.")

    if args.output.exists(): 
        print(f"Warning: Output file {args.output} already exists. It will be overwritten.")
        args.output.unlink()
    try:
        with open(args.input_file, 'r') as f:
            json_data = json.load(f)
        
        convert_json_to_si(json_data, args.output)
        print(f"Successfully converted {args.input_file} to SI units")
        
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found")
        status = 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input_file}")
        status = 1
    except Exception as e:
        print(f"Error: {str(e)}")
        status = 1

    print(args.output.read_text())

    exit(status)
