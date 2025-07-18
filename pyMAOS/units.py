"""
Units handling module for pyMAOS package.

This module provides standardized unit definitions and conversion functions
for structural analysis calculations. It ensures consistent use of units
throughout the program and enables user-defined input/output unit preferences.
"""

import json
import re
import pint
import jsonschema
from pathlib import Path
from jsonschema import validators

# Set up unit registry
ureg = pint.UnitRegistry()
ureg.default_system = 'mks'  # meter-kilogram-second
Q_ = ureg.Quantity  # Shorthand for creating quantities

print(ureg.sys)  # Shows available systems
print(ureg.get_system('imperial'))  # Shows units in the imperial system


# Common imperial units as constants
FOOT = ureg.foot
INCH = ureg.inch
POUND_FORCE = ureg.pound_force
PSI = ureg.psi
KSI = ureg.ksi

# Example predefined imperial display units
IMPERIAL_DISPLAY_UNITS = {
    'force': 'kip',  # 1 kip = 1000 lbf
    'length': 'ft',
    'time': 's',
    'area': 'ft^2',
    'volume': 'ft^3',
    'moment_of_inertía': 'in^4',
    'density': 'lb/ft^3',
    'moment': 'kip*ft',
    'pressure': 'psi',
    'distributed_load': 'kip/ft',
    'rotation': 'rad',
    'angle': 'deg'
}

# Base internal units (all calculations use SI units internally)
INTERNAL_FORCE_UNIT = 'N'  
INTERNAL_LENGTH_UNIT = 'm'
INTERNAL_TIME_UNIT = 's'

# Derived internal units
INTERNAL_AREA_UNIT = f"{INTERNAL_LENGTH_UNIT}^2"
INTERNAL_VOLUME_UNIT = f"{INTERNAL_LENGTH_UNIT}^3"
INTERNAL_MOMENT_OF_INERTIA_UNIT = f"{INTERNAL_LENGTH_UNIT}^4"
INTERNAL_DENSITY_UNIT = f"kg/{INTERNAL_LENGTH_UNIT}^3"
INTERNAL_MOMENT_UNIT = f"{INTERNAL_FORCE_UNIT}*{INTERNAL_LENGTH_UNIT}"
INTERNAL_PRESSURE_UNIT = 'Pa'  # SI unit name (preferred)
INTERNAL_PRESSURE_UNIT_EXPANDED = f"{INTERNAL_FORCE_UNIT}/{INTERNAL_LENGTH_UNIT}^2"  # Expanded form
INTERNAL_DISTRIBUTED_LOAD_UNIT = f"{INTERNAL_FORCE_UNIT}/{INTERNAL_LENGTH_UNIT}"

# Define display/input units with corresponding internal units
# These will be used for input/output and can be updated from JSON
DISPLAY_UNITS = {
    # Base units
    'force': 'kN',                    # Internal: N
    'length': 'm',                    # Internal: m
    'time': 's',                      # Internal: s
    
    # Derived units
    'area': 'm^2',                    # Internal: m^2
    'volume': 'm^3',                  # Internal: m^3
    'moment_of_inertía': 'm^4',       # Internal: m^4
    'density': 'kg/m^3',              # Internal: kg/m^3
    'moment': 'kN*m',                 # Internal: N*m
    'pressure': 'kN/m^2',             # Internal: N/m^2
    'distributed_load': 'kN/m',       # Internal: N/m
    'rotation': 'rad',                # Internal: rad
    'angle': 'deg'                    # Internal: rad
}

# For backward compatibility, keep individual variables
FORCE_UNIT = DISPLAY_UNITS['force']
LENGTH_UNIT = DISPLAY_UNITS['length']
MOMENT_UNIT = DISPLAY_UNITS['moment']
PRESSURE_UNIT = DISPLAY_UNITS['pressure']
DISTRIBUTED_LOAD_UNIT = DISPLAY_UNITS['distributed_load']

def update_units_from_json(json_string):
    """
    Parse JSON unit definitions and update display/input unit variables.
    
    Parameters
    ----------
    json_string : str
        JSON string containing unit definitions
        
    Returns
    -------
    bool
        True if units were successfully updated, False otherwise
    """
    try:
        # Clean up the JSON string to ensure it's valid
        json_string = json_string.strip()
        if not json_string.startswith('{'):
            return False
        
        # Parse JSON
        unit_dict = json.loads(json_string)
        global DISPLAY_UNITS, FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT
        
        # Update display units based on JSON specification
        if "force" in unit_dict:
            DISPLAY_UNITS['force'] = unit_dict["force"]
        if "length" in unit_dict:
            DISPLAY_UNITS['length'] = unit_dict["length"]
        if "pressure" in unit_dict:
            DISPLAY_UNITS['pressure'] = unit_dict["pressure"]
        
        # Handle special case where distance differs from length
        distance_unit = unit_dict.get("distance", DISPLAY_UNITS['length'])
        
        # Update derived units based on base units
        DISPLAY_UNITS['moment'] = f"{DISPLAY_UNITS['force']}*{DISPLAY_UNITS['length']}"
        DISPLAY_UNITS['distributed_load'] = f"{DISPLAY_UNITS['force']}/{distance_unit}"
        DISPLAY_UNITS['area'] = f"{DISPLAY_UNITS['length']}^2"
        DISPLAY_UNITS['volume'] = f"{DISPLAY_UNITS['length']}^3"
        DISPLAY_UNITS['moment_of_inertía'] = f"{DISPLAY_UNITS['length']}^4"
        
        # Update individual variables for backward compatibility
        FORCE_UNIT = DISPLAY_UNITS['force']
        LENGTH_UNIT = DISPLAY_UNITS['length']
        MOMENT_UNIT = DISPLAY_UNITS['moment']
        PRESSURE_UNIT = DISPLAY_UNITS['pressure'] 
        DISTRIBUTED_LOAD_UNIT = DISPLAY_UNITS['distributed_load']
        
        print(f"Using display units from JSON: Force={FORCE_UNIT}, Length={LENGTH_UNIT}, Moment={MOMENT_UNIT}")
        print(f"Pressure={PRESSURE_UNIT}, Distributed Load={DISTRIBUTED_LOAD_UNIT}")
        print(f"Note: All internal calculations will use SI units.")
        return True
    except Exception as e:
        print(f"Error updating units: {e}")
        return False

def parse_value_with_units(value_string):
    """
    Parse a string that may contain a value with units without spaces between.
    
    Parameters
    ----------
    value_string : str
        String containing a value and potentially unit information
        
    Returns
    -------
    float or pint.Quantity
        Parsed value, either as a dimensionless float or with units
    
    Examples
    --------
    >>> parse_value_with_units("10kN")
    <Quantity(10, 'kilonewton')>
    >>> parse_value_with_units("30.5")
    30.5
    >>> parse_value_with_units("2.54cm")
    <Quantity(2.54, 'centimeter')>
    """
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

def convert_to_internal_units(value, unit_type):
    """
    Convert a value from display units to internal (SI) units.
    
    Parameters
    ----------
    value : float or pint.Quantity
        The value to convert
    unit_type : str
        Type of unit (e.g., 'force', 'length', 'moment')
        
    Returns
    -------
    float
        The value in internal units
    """
    if isinstance(value, pint.Quantity):
        internal_unit = get_internal_unit(unit_type)
        return value.to(internal_unit).magnitude
    return value

def convert_to_display_units(value, unit_type):
    """
    Convert a value from internal (SI) units to display units.
    
    Parameters
    ----------
    value : float
        The value in internal units
    unit_type : str
        Type of unit (e.g., 'force', 'length', 'moment')
        
    Returns
    -------
    float
        The value in display units
    """
    internal_unit = get_internal_unit(unit_type)
    display_unit = DISPLAY_UNITS.get(unit_type)
    if display_unit and internal_unit:
        try:
            return Q_(value, internal_unit).to(display_unit).magnitude
        except:
            pass
    return value

def get_internal_unit(unit_type):
    """Get the internal unit for a specific quantity type"""
    mapping = {
        'force': INTERNAL_FORCE_UNIT,
        'length': INTERNAL_LENGTH_UNIT,
        'time': INTERNAL_TIME_UNIT,
        'area': INTERNAL_AREA_UNIT,
        'volume': INTERNAL_VOLUME_UNIT,
        'moment_of_inertía': INTERNAL_MOMENT_OF_INERTIA_UNIT,
        'density': INTERNAL_DENSITY_UNIT,
        'moment': INTERNAL_MOMENT_UNIT,
        'pressure': INTERNAL_PRESSURE_UNIT,
        'distributed_load': INTERNAL_DISTRIBUTED_LOAD_UNIT
    }
    return mapping.get(unit_type)

def extend_with_dimension_validator(validator_class):
    """Extend JSON Schema validator with dimension validation"""
    validate_properties = validator_class.VALIDATORS["properties"]
    
    def set_dimensions(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if instance is not None and property in instance:
                # Process normal validation
                for error in validate_properties(validator, properties, instance, schema):
                    yield error
                
                # Add dimension validation if specified
                if "dimension" in subschema and instance[property] is not None:
                    dimension = subschema["dimension"]
                    value = instance[property]
                    
                    # Handle string values with units
                    if isinstance(value, str):
                        try:
                            # Parse and validate the unit dimensions
                            parsed = parse_value_with_units(value)
                            if isinstance(parsed, pint.Quantity):
                                # Get the expected dimension's internal unit
                                expected_unit = get_internal_unit(dimension)
                                if expected_unit:
                                    try:
                                        # Try converting - will fail if dimensions don't match
                                        parsed.to(expected_unit)
                                    except pint.DimensionalityError as e:
                                        yield jsonschema.exceptions.ValidationError(
                                            f"Value '{value}' has incorrect dimension for '{property}'. "
                                            f"Expected {dimension} ({expected_unit}), got {parsed.units}")
                        except Exception as e:
                            yield jsonschema.exceptions.ValidationError(
                                f"Failed to parse unit dimensions for '{property}': {str(e)}")
    
    return validators.extend(validator_class, {"properties": set_dimensions})

# Create our custom validator
DimensionValidator = extend_with_dimension_validator(jsonschema.Draft7Validator)

def validate_input_with_schema(input_file, schema_file=None):
    """
    Validate a JSON input file against the schema with dimension validation
    
    Parameters
    ----------
    input_file : str
        Path to the input JSON file to validate
    schema_file : str, optional
        Path to the schema file (defaults to data/myschema.json)
        
    Returns
    -------
    bool
        True if validation succeeds
        
    Raises
    ------
    ValidationError
        If validation fails
    """
    # Default schema location
    if schema_file is None:
        # Try to find schema in a few common locations
        possible_paths = [
            Path("data/myschema.json"),
            Path("myschema.json"),
            Path(__file__).parent.parent / "data" / "myschema.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                schema_file = str(path)
                break
        else:
            raise FileNotFoundError("Could not find schema file 'myschema.json'")
    
    # Load the schema
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    
    # Load the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create validator instance
    validator = DimensionValidator(schema)
    
    # Collect and report all errors
    errors = list(validator.iter_errors(data))
    if errors:
        print(f"Validation failed with {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            # Format the error path as a JSON path
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            print(f"{i}. Error at '{path}': {error.message}")
        
        # Raise the first error to stop execution if needed
        raise jsonschema.exceptions.ValidationError(f"Validation failed: {errors[0].message}")
    
    print("Input file successfully validated against schema!")
    return True