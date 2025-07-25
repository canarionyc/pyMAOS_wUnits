"""
Units handling module for pyMAOS package.

This module provides standardized unit definitions and conversion functions
for structural analysis calculations. It ensures consistent use of units
throughout the program and enables user-defined input/output unit preferences.
"""
import re
import pint

from pprint import pprint

global SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS
from pint import UnitRegistry, Quantity
# Predefined unit systems
SI_UNITS = {
    "force": "N",
    "length": "m",
    "pressure": "Pa",
    "distance": "m",
    "moment": "N*m",
    "distributed_load": "N/m",
    "area": "m^2",
    "moment_of_inertia": "m^4",
    "density": "kg/m^3"
}

IMPERIAL_UNITS = {
    "force": "klbf",
    "length": "in",
    "pressure": "ksi",
    "distance": "ft",
    "moment": "klbf * foot",
    "distributed_load": "klbf / foot",
    "area": "in^2",
    "moment_of_inertia": "in^4",
    "density": "klb/in^3"
}

METRIC_KN_UNITS = {
    "force": "kN",
    "length": "cm",
    "pressure": "kN/m^2",
    "distance": "m",
    "moment": "kN*m",
    "distributed_load": "kN/m",
    "area": "cm^2",
    "moment_of_inertia": "cm^4",
    "density": "kg/cm^3"
}
pprint(globals()); print(dir())
# Set up unit registry
ureg = pint.UnitRegistry()
# ureg.default_system = 'mks'  # meter-kilogram-second
Q_ = ureg.Quantity  # Shorthand for creating quantities

print(ureg.sys)  # Shows available systems
print(ureg.get_system('imperial'))  # Shows units in the imperial system

print(ureg.get_system('imperial').units)  # Shows the units in the imperial system
print(str(ureg.get_system('imperial')))   # String representation
print(ureg.get_system('imperial').name)   # System name

pprint(ureg.get_system('imperial').units)

global FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY, LENGTH_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
# Add these definitions to the top of the file after initializing ureg
# Pre-computed dimensionality constants for efficient type checking
FORCE_DIMENSIONALITY = ureg.N.dimensionality
MOMENT_DIMENSIONALITY = (ureg.N * ureg.m).dimensionality
LENGTH_DIMENSIONALITY = ureg.m.dimensionality
PRESSURE_DIMENSIONALITY = ureg.Pa.dimensionality
DISTRIBUTED_LOAD_DIMENSIONALITY = (ureg.N / ureg.m).dimensionality

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

global INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_TIME_UNIT
# Base internal units (all calculations use SI units)
INTERNAL_FORCE_UNIT = 'N'  
INTERNAL_LENGTH_UNIT = 'm'
INTERNAL_TIME_UNIT = 's'

# Derived internal units
global INTERNAL_AREA_UNIT, INTERNAL_VOLUME_UNIT, INTERNAL_MOMENT_OF_INERTIA_UNIT, INTERNAL_DENSITY_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED, INTERNAL_DISTRIBUTED_LOAD_UNIT
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
FORCE_DISPLAY_UNIT = DISPLAY_UNITS['force']
LENGTH_DISPLAY_UNIT = DISPLAY_UNITS['length']
MOMENT_DISPLAY_UNIT = DISPLAY_UNITS['moment']
PRESSURE_DISPLAY_UNIT = DISPLAY_UNITS['pressure']
DISTRIBUTED_LOAD_DISPLAY_UNIT = DISPLAY_UNITS['distributed_load']

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
        import json
        unit_dict = json.loads(json_string)
        global DISPLAY_UNITS, FORCE_DISPLAY_UNIT, LENGTH_DISPLAY_UNIT, MOMENT_DISPLAY_UNIT, PRESSURE_DISPLAY_UNIT, DISTRIBUTED_LOAD_DISPLAY_UNIT
        
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
        MOMENT_DISPLAY_UNIT = DISPLAY_UNITS['moment']
        PRESSURE_DISPLAY_UNIT = DISPLAY_UNITS['pressure']
        DISTRIBUTED_LOAD_UNIT = DISPLAY_UNITS['distributed_load']
        
        print(f"Using display units from JSON: Force={FORCE_UNIT}, Length={LENGTH_UNIT}, Moment={MOMENT_DISPLAY_UNIT}")
        print(f"Pressure={PRESSURE_DISPLAY_UNIT}, Distributed Load={DISTRIBUTED_LOAD_UNIT}")
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



class UnitManager:
    """
    Central manager for unit system handling throughout pyMAOS
    
    This singleton class provides a consistent interface for all unit operations
    and ensures that unit settings are synchronized across the package.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnitManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Default to SI units
        self.current_system = SI_UNITS
        self.system_name = "SI"
        
        # Initialize registry once
        self.ureg = pint.UnitRegistry()
        
        # Keep track of all registered modules that need unit updates
        self.registered_modules = []
    
    def set_unit_system(self, system_dict, system_name=None):
        """
        Set the active unit system for the entire package
        
        Parameters
        ----------
        system_dict : dict
            Dictionary mapping dimension names to unit strings
        system_name : str, optional
            Name of the unit system (e.g., "SI", "imperial", "metric_kn")
        """
        self.current_system = system_dict
        self.system_name = system_name or "custom"
        from pprint import pp, pprint; pprint(globals())
        # Update global unit variables
        global DISPLAY_UNITS, FORCE_DISPLAY_UNIT, LENGTH_DISPLAY_UNIT, MOMENT_DISPLAY_UNIT, PRESSURE_DISPLAY_UNIT, DISTRIBUTED_LOAD_UNIT
        pprint(globals())
        DISPLAY_UNITS = system_dict
        FORCE_DISPLAY_UNIT = system_dict.get("force", "kN")
        LENGTH_DISPLAY_UNIT = system_dict.get("length", "m")
        MOMENT_DISPLAY_UNIT = system_dict.get("moment", "kN*m")
        PRESSURE_DISPLAY_UNIT = system_dict.get("pressure", "GPa")
        DISTRIBUTED_LOAD_UNIT = system_dict.get("distributed_load", "kN/m")
        
        # Notify all registered modules of unit change
        for module_update_func in self.registered_modules:
            try:
                module_update_func(system_dict)
            except Exception as e:
                print(f"Error updating module with new units: {e}")
    
    def register_for_unit_updates(self, update_function):
        """Register a module to receive unit system updates"""
        self.registered_modules.append(update_function)
    
    def get_current_units(self):
        """Get the current unit system dictionary"""
        return self.current_system
        
    def get_system_name(self):
        """Get the name of the current unit system"""
        return self.system_name
    
    def convert_value(self, value, from_unit, to_unit):
        """
        Convert a value from one unit to another
        
        Parameters
        ----------
        value : float
            Value to convert
        from_unit : str
            Source unit (e.g., "N")
        to_unit : str
            Target unit (e.g., "lbf")
            
        Returns
        -------
        float
            Converted value
        """
        if from_unit == to_unit:
            return value
            
        try:
            quantity = self.ureg.Quantity(float(value), from_unit)
            return quantity.to(to_unit).magnitude
        except Exception as e:
            print(f"Warning: Unit conversion failed from {from_unit} to {to_unit}: {e}")
            return value

# Create the singleton instance
unit_manager = UnitManager()

# Replace existing set_unit_system function to use the manager
def set_unit_system(system_dict, system_name=None):
    """Set the unit system throughout the package"""
    unit_manager.set_unit_system(system_dict, system_name)

def convert_to_unit_system(array, target_system_name):
    """
    Convert all pint.Quantity objects in a numpy array to units from a specified unit system.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing pint.Quantity objects.
    target_system_name : str
        Name of the unit system to convert to ('SI', 'imperial', or 'metric_kn').

    Returns
    -------
    numpy.ndarray
        Array with magnitudes in the target system units.
    """
    # Get the target unit system dictionary
    if target_system_name.lower() == 'si':
        target_system = SI_UNITS
    elif target_system_name.lower() == 'imperial':
        target_system = IMPERIAL_UNITS
    elif target_system_name.lower() in ('metric_kn', 'metric'):
        target_system = METRIC_KN_UNITS
    else:
        target_system = unit_manager.get_current_units()

    # Create dimensionality to unit type mapping
    dim_to_unit_type = {
        FORCE_DIMENSIONALITY: 'force',
        LENGTH_DIMENSIONALITY: 'length',
        PRESSURE_DIMENSIONALITY: 'pressure',
        MOMENT_DIMENSIONALITY: 'moment',
        DISTRIBUTED_LOAD_DIMENSIONALITY: 'distributed_load'
    }

    # Initialize result array
    converted_array = np.empty_like(array, dtype=float)

    # Convert each element
    for index, value in np.ndenumerate(array):
        if isinstance(value, pint.Quantity):
            # Get dimensionality of the quantity
            dim = value.dimensionality

            # Find matching unit type
            unit_type = None
            for dim_key, type_name in dim_to_unit_type.items():
                if dim == dim_key:
                    unit_type = type_name
                    break

            if unit_type and unit_type in target_system:
                # Convert to the target unit
                target_unit = target_system[unit_type]
                converted_array[index] = value.to(target_unit).magnitude
            else:
                # If we can't determine the unit type, keep the magnitude
                converted_array[index] = value.magnitude
        else:
            # If it's not a Quantity, keep it as is
            converted_array[index] = value

    return converted_array

if __name__ == "__main__":
    # Example usage
    set_unit_system(IMPERIAL_UNITS, "imperial")
    print("Current unit system:", unit_manager.get_system_name())

    # Convert a value
    value = Q_(10, 'N')
    converted_value = unit_manager.convert_value(value.magnitude, 'N', 'lbf')
    print(f"Converted {value} to {converted_value} lbf")

    # Convert a numpy array
    import numpy as np
    arr = np.array([Q_(10, 'N'), Q_(20, 'm')])
    converted_arr = convert_to_unit_system(arr, 'imperial')
    print("Converted array:", converted_arr)

    # Example usage
    array_with_units = np.array([
        ureg.Quantity(5, 'meter'),
        ureg.Quantity(10, 'newton')
    ], dtype=object)

    # Convert to imperial units
    imperial_values = convert_to_unit_system(array_with_units, 'imperial')
    print(imperial_values)  # Values in inches and klbf
