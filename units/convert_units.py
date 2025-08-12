"""
Convert JSON structural model from any unit system to SI units.
"""
import os
import json
import re
import sys
import argparse
import pint

# Initialize unit registry
from pyMAOS import unit_manager

Q_ = unit_manager.ureg.Quantity

# Define dimension mappings - what fields have what physical dimensions
DIMENSION_REGISTRY = {
    "nodes": {
        "x": "length",
        "y": "length",
        "z": "length"
    },
    "materials": {
        "E": "pressure",
        "G": "pressure",
        "rho": "density",
        "alpha": "thermal_expansion"
    },
    "sections": {
        "area": "area",
        "r": "length",
        "ixx": "moment_of_inertia"
    },
    "joint_loads": {
        "fx": "force",
        "fy": "force",
        "fz": "force",
        "mx": "moment",
        "my": "moment",
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
    "rotation": "rad",
    "density": "kg/m^3",
    "thermal_expansion": "1/K"
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

def format_value_with_units(value, unit):
    """Format a numeric value with its unit as a string."""
    # Handle special formatting for different types of values
    if isinstance(value, float):
        # Use appropriate precision based on magnitude
        if abs(value) < 0.001:
            formatted = f"{value:.6e}"
        elif abs(value) < 1:
            formatted = f"{value:.6f}"
        elif abs(value) < 1000:
            formatted = f"{value:.4f}"
        else:
            formatted = f"{value:.2f}"

        # Remove trailing zeros and decimal point if no fractional part
        formatted = formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

    else:
        formatted = str(value)

    # Combine value with unit
    return f"{formatted} {unit}".strip()

def convert_to_si(value, dimension, unit_system, default_units, as_string=True):
    """
    Convert a value to SI units based on its dimension.

    Parameters
    ----------
    value : str, float, int, or pint.Quantity
        The value to convert
    dimension : str
        The physical dimension of the value
    unit_system : str
        The unit system of the input value
    default_units : dict
        Dictionary mapping dimensions to units for the input
    as_string : bool, optional
        Whether to return the result as a string with units (True) or numeric value (False)

    Returns
    -------
    str or float
        The converted value, either as a string with units or as a numeric value
    """
    si_unit = SI_UNITS.get(dimension, "")

    try:
        if isinstance(value, pint.Quantity):
            # Value already has explicit units
            if si_unit:
                si_value = value.to(si_unit).magnitude
                return format_value_with_units(si_value, si_unit) if as_string else si_value
        elif isinstance(value, (int, float)) and dimension in default_units:
            # Value has implicit units based on default_units
            implicit_unit = default_units[dimension]
            quantity = Q_(value, implicit_unit)
            if si_unit:
                si_value = quantity.to(si_unit).magnitude
                return format_value_with_units(si_value, si_unit) if as_string else si_value
        elif isinstance(value, str):
            # Try parsing the string as a value with units
            parsed = parse_value_with_units(value)
            if isinstance(parsed, pint.Quantity) and si_unit:
                si_value = parsed.to(si_unit).magnitude
                return format_value_with_units(si_value, si_unit) if as_string else si_value
            elif isinstance(parsed, (int, float)) and dimension in default_units:
                # Parsed value was numeric, use default units
                implicit_unit = default_units[dimension]
                quantity = Q_(parsed, implicit_unit)
                if si_unit:
                    si_value = quantity.to(si_unit).magnitude
                    return format_value_with_units(si_value, si_unit) if as_string else si_value
    except Exception as e:
        print(f"Warning: Error converting '{value}' for dimension '{dimension}': {e}")

    # No conversion needed or possible - return original value
    return value

def convert_json_to_si(json_data, output_file=None, as_string=True):
    """
    Convert all values in a JSON structure to SI units.

    Parameters
    ----------
    json_data : dict
        The JSON data to convert
    output_file : str, optional
        Path to output file
    as_string : bool, optional
        Whether to output quantities as strings with units (True) or raw numeric values (False)

    Returns
    -------
    dict
        The converted data
    """
    # Extract unit system and default units
    unit_system = json_data.get("unit_system", "SI").lower()
    default_units = json_data.get("units", {})

    print(f"Converting from {unit_system} system to SI units")
    print(f"Output format: {'String with units' if as_string else 'Numeric values'}")

    # Set up output data structure
    si_data = json_data.copy()

    # Replace unit system
    si_data["unit_system"] = "SI"

    # Update units section to SI
    si_data["units"] = {
        "length": "m",
        "pressure": "Pa",
        "force": "N",
        "area": "m^2",
        "moment_of_inertia": "m^4",
        "distributed_load": "N/m",
        "angle": "rad",
        "density": "kg/m^3"
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

                            # Convert the value with the appropriate output format
                            si_data[section_name][i][field] = convert_to_si(
                                original_value, dimension, unit_system, default_units, as_string=as_string)

                            print(f"Converted {section_name}[{i}].{field} to SI: {si_data[section_name][i][field]}")

    # Write to output file or print to stdout
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(si_data, f, indent=2)
        print(f"Converted data written to {output_file}")
    else:
        print(json.dumps(si_data, indent=2))

    return si_data

def convert_si_to_display_units(si_data, output_file=None, display_units=None, as_string=True):
    """
    Convert structural analysis results from SI units to display units

    Parameters
    ----------
    si_data : dict or str
        JSON data structure or path to JSON file with SI units
    output_file : str, optional
        Path to output file (defaults to None, which returns converted data)
    display_units : dict, optional
        Dictionary mapping dimensions to display units
    as_string : bool, optional
        Whether to output quantities as strings with units (True) or raw numeric values (False)

    Returns
    -------
    dict
        Data structure with values converted to display units
    """
    import json
    from copy import deepcopy

    # Handle case where si_data is a file path
    if isinstance(si_data, str):
        with open(si_data, 'r') as f:
            si_data = json.load(f)

    # Get display units if not provided
    if display_units is None:
        try:
            from pyMAOS.units_mod import DISPLAY_UNITS
            display_units = DISPLAY_UNITS
        except ImportError:
            # Fallback to some standard display units
            display_units = {
                "force": "lbf",
                "length": "ft",
                "moment": "lbf*ft",
                "pressure": "psi",
                "area": "in^2",
                "moment_of_inertia": "in^4",
                "distributed_load": "lbf/ft",
                "rotation": "rad",
                "density": "lb/ft^3",
                "thermal_expansion": "1/F"
            }

    # Define SI units for each dimension
    si_units = {
        "force": "N",
        "length": "m",
        "moment": "N*m",
        "pressure": "Pa",
        "area": "m^2",
        "moment_of_inertia": "m^4",
        "distributed_load": "N/m",
        "rotation": "rad",
        "density": "kg/m^3",
        "thermal_expansion": "1/K"
    }

    # Create a copy of the data to modify
    display_data = deepcopy(si_data)

    # Update the units section
    if "units" in display_data:
        display_data["units"] = display_units

    # Convert a single value using pint and return as string with units or numeric value
    def convert_value_with_format(value, from_unit, to_unit, as_string=True):
        try:
            # Handle string with units
            if isinstance(value, str):
                parsed = parse_value_with_units(value)
                if isinstance(parsed, pint.Quantity):
                    value = parsed.magnitude
                    from_unit = str(parsed.units)
                else:
                    value = parsed  # Use the parsed numeric value

            # Skip conversion for values very close to zero
            if isinstance(value, (int, float)) and abs(float(value)) < 1e-12:
                return format_value_with_units(0.0, to_unit) if as_string else 0.0

            # Convert using pint
            quantity = Q_(float(value), from_unit)
            converted = quantity.to(to_unit).magnitude

            return format_value_with_units(converted, to_unit) if as_string else converted
        except Exception as e:
            print(f"Error converting {value} from {from_unit} to {to_unit}: {e}")
            return value

    # Process dimensions in node data
    for section_name, field_dimensions in DIMENSION_REGISTRY.items():
        if section_name in si_data:
            section_data = si_data[section_name]

            # Handle array sections
            if isinstance(section_data, list):
                for i, item in enumerate(section_data):
                    for field, dimension in field_dimensions.items():
                        if field in item:
                            # Get the appropriate SI and display units for this dimension
                            from_unit = si_units.get(dimension, "")
                            to_unit = display_units.get(dimension, from_unit)

                            # Convert the value
                            original_value = item[field]
                            display_data[section_name][i][field] = convert_value_with_format(
                                original_value, from_unit, to_unit, as_string=as_string)

    # Write to output file if provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(display_data, f, indent=2)
        print(f"Converted results written to {output_file}")

    return display_data

if __name__ == "__main__":
    status = 0
    """Command line interface for unit conversion."""
    parser = argparse.ArgumentParser(description="Convert JSON structural model to SI units")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    parser.add_argument("--output_type", choices=["string", "numeric"], default="string",
                      help="Format of output quantities: 'string' includes units, 'numeric' for raw values")
    args = parser.parse_args()

    # Determine output filename without using Path module
    if not args.output:
        input_basename = os.path.basename(args.input_file)
        input_name, input_ext = os.path.splitext(input_basename)

        # Remove "_imperial" from input name if present
        if "_imperial" in input_name:
            input_name = input_name.replace("_imperial", "")

        # Add appropriate suffix based on output type
        output_type_suffix = "_SI_string" if args.output_type == "string" else "_SI_numeric"
        output_basename = f"{input_name}{output_type_suffix}{input_ext}"

        args.output = os.path.join(os.path.dirname(args.input_file), output_basename)

    if args.output == args.input_file:
        print(f"Warning: Output file {args.output} is the same as input file {args.input_file}.")

    if os.path.exists(args.output):
        print(f"Warning: Output file {args.output} already exists. It will be overwritten.")
        os.remove(args.output)

    try:
        print(f"Converting {args.input_file} to SI units with {args.output_type} format")
        with open(args.input_file, 'r') as f:
            json_data = json.load(f)

        # Convert to SI with appropriate output format
        as_string = args.output_type == "string"
        convert_json_to_si(json_data, args.output, as_string=as_string)
        print(f"Successfully converted {args.input_file} to SI units")

        # Print the contents of the output file
        with open(args.output, 'r') as f:
            print(f.read())

    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found")
        status = 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input_file}")
        status = 1
    except Exception as e:
        print(f"Error: {str(e)}")
        status = 1

    sys.exit(status)
