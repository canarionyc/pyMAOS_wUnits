"""
Units handling module for pyMAOS package.

This module provides standardized unit definitions and conversion functions
for structural analysis calculations. It ensures consistent use of units
throughout the program and enables user-defined input/output unit preferences.
"""
import re
import pint
import numpy as np
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
# pprint(globals())
# print(dir())
# Set up unit registry
# ureg = pint.UnitRegistry()
# ureg.default_system = 'mks'  # meter-kilogram-second
# Q_ = ureg.Quantity  # Shorthand for creating quantities
#
# print(ureg.sys)  # Shows available systems
# print(ureg.get_system('imperial'))  # Shows units in the imperial system
#
# print(ureg.get_system('imperial').units)  # Shows the units in the imperial system
# print(str(ureg.get_system('imperial')))   # String representation
# print(ureg.get_system('imperial').name)   # System name
#
# pprint(ureg.get_system('imperial').units)


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


class UnitManager:
    """
    Central manager for unit system handling throughout pyMAOS

    This singleton class provides a consistent interface for all unit operations
    and ensures that unit settings are synchronized across the package.
    """
    _instance = None
    _initialized = False

    def __new__(cls, internal_system="SI"):
        if cls._instance is None:
            cls._instance = super(UnitManager, cls).__new__(cls)
            cls._instance._initialize(internal_system)
        return cls._instance

    def _initialize(self, internal_system="SI"):
        """Initialize the unit manager with specified internal unit system"""
        if self._initialized:
            print("WARNING: UnitManager already initialized - ignoring internal system change request")
            return

        # Initialize registry once
        self.ureg = pint.UnitRegistry()

        # Keep track of all registered modules that need unit updates
        self.registered_modules = []

        # Default display units to SI
        self.current_system = SI_UNITS
        self.system_name = "SI"

        # Set internal units based on specified system
        if internal_system.lower() == "imperial":
            print("DEBUG: Initializing UnitManager with IMPERIAL internal units")
            # Internal base units for Imperial system
            self.INTERNAL_FORCE_UNIT = 'lbf'
            self.INTERNAL_LENGTH_UNIT = 'inch'
            self.INTERNAL_TIME_UNIT = 's'
            self.INTERNAL_ROTATION_UNIT = 'rad'
        else:
            print("DEBUG: Initializing UnitManager with SI internal units")
            # Internal base units for SI system (default)
            self.INTERNAL_FORCE_UNIT = 'N'
            self.INTERNAL_LENGTH_UNIT = 'm'
            self.INTERNAL_TIME_UNIT = 's'
            self.INTERNAL_ROTATION_UNIT = 'rad'

        # Calculate derived units based on base units
        self._update_derived_units()

        # Mark as initialized
        self._initialized = True
        print(f"DEBUG: UnitManager initialized with internal {internal_system} units")

    # Rest of the class remains the same...

    def _update_derived_units(self):
        """Update all derived internal units based on current base units"""
        # Derived internal units
        self.INTERNAL_AREA_UNIT = f"{self.INTERNAL_LENGTH_UNIT}^2"
        self.INTERNAL_VOLUME_UNIT = f"{self.INTERNAL_LENGTH_UNIT}^3"
        self.INTERNAL_MOMENT_OF_INERTIA_UNIT = f"{self.INTERNAL_LENGTH_UNIT}^4"
        self.INTERNAL_DENSITY_UNIT = f"kg/{self.INTERNAL_LENGTH_UNIT}^3"
        self.INTERNAL_MOMENT_UNIT = f"{self.INTERNAL_FORCE_UNIT}*{self.INTERNAL_LENGTH_UNIT}"
        self.INTERNAL_PRESSURE_UNIT = 'Pa'  # SI unit name (preferred)
        self.INTERNAL_PRESSURE_UNIT_EXPANDED = f"{self.INTERNAL_FORCE_UNIT}/{self.INTERNAL_LENGTH_UNIT}^2"
        self.INTERNAL_DISTRIBUTED_LOAD_UNIT = f"{self.INTERNAL_FORCE_UNIT}/{self.INTERNAL_LENGTH_UNIT}"

    def set_internal_units(self, force=None, length=None, time=None, rotation=None):
        """
        Update the internal base units and recalculate derived units

        Parameters
        ----------
        force : str, optional
            Base force unit
        length : str, optional
            Base length unit
        time : str, optional
            Base time unit
        rotation : str, optional
            Base rotation unit
        """
        if force is not None:
            self.INTERNAL_FORCE_UNIT = force
        if length is not None:
            self.INTERNAL_LENGTH_UNIT = length
        if time is not None:
            self.INTERNAL_TIME_UNIT = time
        if rotation is not None:
            self.INTERNAL_ROTATION_UNIT = rotation

        # Update derived units based on new base units
        self._update_derived_units()
        print(
            f"DEBUG: Internal base units updated - Force: {self.INTERNAL_FORCE_UNIT}, Length: {self.INTERNAL_LENGTH_UNIT}")

    def get_internal_unit(self, unit_type):
        """
        Get the internal unit for a specific quantity type

        Parameters
        ----------
        unit_type : str
            Type of unit (e.g., 'force', 'length', 'moment')

        Returns
        -------
        str
            The internal unit string for the requested quantity type
        """
        mapping = {
            'force': self.INTERNAL_FORCE_UNIT,
            'length': self.INTERNAL_LENGTH_UNIT,
            'time': self.INTERNAL_TIME_UNIT,
            'rotation': self.INTERNAL_ROTATION_UNIT,
            'area': self.INTERNAL_AREA_UNIT,
            'volume': self.INTERNAL_VOLUME_UNIT,
            'moment_of_inertia': self.INTERNAL_MOMENT_OF_INERTIA_UNIT,
            'density': self.INTERNAL_DENSITY_UNIT,
            'moment': self.INTERNAL_MOMENT_UNIT,
            'pressure': self.INTERNAL_PRESSURE_UNIT,
            'distributed_load': self.INTERNAL_DISTRIBUTED_LOAD_UNIT
        }
        return mapping.get(unit_type)

    def set_display_unit_system(self, system_dict, system_name=None):
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
        self.ureg.default_preferred_units = IMPERIAL_UNITS
        self.system_name = system_name or "custom"
        from pprint import pp, pprint
        pprint(globals())
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

    def parse_value(self, value_string):
        """
        Parse a string that may contain a value with units using the manager's registry.

        Parameters
        ----------
        value_string : str
            String containing a value and potentially unit information

        Returns
        -------
        float or pint.Quantity
            Parsed value, either as a dimensionless float or with units
        """
        # Match pattern: [numeric value][units]
        # The numeric part can include scientific notation like 30.0e6
        match = re.match(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(.*)', value_string.strip())

        if match:
            value_str, unit_str = match.groups()
            value = float(value_str)

            if unit_str and unit_str.strip():
                try:
                    # Create quantity with UnitManager's registry
                    value_with_units = self.ureg.Quantity(value, unit_str)
                    return value_with_units
                except Exception as e:
                    print(f"Warning: Could not parse unit '{unit_str}', treating as dimensionless. Error: {e}")
                    return value
            return value

        # If no match, try to evaluate as a simple numeric expression
        try:
            return float(eval(value_string))
        except Exception as e:
            raise ValueError(f"Could not parse value: {value_string}. Error: {e}")

    def get_zero_quantity(self, unit_str):
        """
        Creates a zero quantity with the specified unit.

        Parameters
        ----------
        unit_str : str
            The unit string (e.g., 'N', 'kN', 'lb')

        Returns
        -------
        pint.Quantity
            A quantity object with value 0 and the specified unit
        """
        # Parse the unit string to get a valid unit object
        try:
            # For simple units
            unit = self.ureg.parse_units(unit_str)

            # Create and return a zero quantity with this unit
            return self.ureg.Quantity(0, unit)
        except Exception as e:
            print(f"DEBUG: Error creating zero quantity with unit '{unit_str}': {e}")
            # Fallback to dimensionless
            return self.ureg.Quantity(0, '')
    def get_conjugate_dimension(self, dimension):
        """
        Get the work conjugate dimension for a given physical dimension.

        Parameters
        ----------
        dimension : str
            The dimension string like '[force]' or '[length]'

        Returns
        -------
        str
            The conjugate dimension string
        """
        return WORK_CONJUGATE_DIMENSIONS.get(dimension, dimension)

    def get_conjugate_unit(self, unit):
            """
            Get the work conjugate unit for a given unit.

            Parameters
            ----------
            unit : pint.Unit or str or None
                The unit whose conjugate is desired

            Returns
            -------
            str
                The conjugate unit string
            """
            # Handle None values
            if unit is None:
                print(f"DEBUG: Received None unit in get_conjugate_unit, returning 'dimensionless'")
                return 'dimensionless'

            # Debug output
            print(f"Finding conjugate for unit: {unit}")

            # Special case for radians (as string)
            if isinstance(unit, str) and (unit == 'rad' or unit == 'radian' or unit == 'radians'):
                print(f"DEBUG: Angular unit '{unit}' detected - conjugate is moment")
                return self.get_current_units().get('moment', 'N*m')

            # Convert string to unit if needed
            if isinstance(unit, str):
                unit = self.ureg.parse_units(unit)

            # Get dimension string
            dim_str = str(unit.dimensionality)
            print(f"Original dimension string: {dim_str}")

            # Special case for angular units (radians are dimensionless in pint but need special handling)
            if dim_str == 'dimensionless' and str(unit) in ['radian', 'rad']:
                print(f"DEBUG: Radian unit detected - conjugate is moment")
                return self.get_current_units().get('moment', 'N*m')

            # Normalize common physical quantities to their canonical dimension names
            # Map standard physical quantities to their dimension types
            dimension_map = {
                # Force: [mass] * [length] / [time]^2
                "[mass] * [length] / [time] ** 2": "[force]",
                "[mass] * [length] * [time] ** -2": "[force]",

                # Moment: [mass] * [length]^2 / [time]^2
                "[mass] * [length] ** 2 / [time] ** 2": "[moment]",
                "[mass] * [length] ** 2 * [time] ** -2": "[moment]",

                # Pressure: [mass] / ([length] * [time]^2)
                "[mass] / ([length] * [time] ** 2)": "[pressure]",
                "[mass] * [length] ** -1 * [time] ** -2": "[pressure]"
            }

            # Try to match the dimension string to a known physical quantity
            normalized_dim = dimension_map.get(dim_str, dim_str)
            print(f"Normalized dimension: {normalized_dim}")

            # Now find the conjugate using the normalized dimension
            conj_dim = self.get_conjugate_dimension(normalized_dim)
            print(f"Conjugate dimension: {conj_dim}")

            # Return base unit for that dimension
            base_unit = DIMENSION_TO_BASE_UNIT.get(conj_dim, 'dimensionless')
            print(f"Base unit for conjugate: {base_unit}")

            # Handle units with combined dimensions (like moment = force * length)
            if "force" in str(unit.dimensionality) and "length" in str(unit.dimensionality):
                if base_unit == 'm':
                    # If force*length conjugates to length, return length unit
                    return self.get_current_units().get('length', 'm')

            # Return appropriate unit from current unit system if possible
            if conj_dim == '[length]':
                return self.get_current_units().get('length', 'm')
            elif conj_dim == '[force]':
                return self.get_current_units().get('force', 'N')
            elif conj_dim == '[moment]':
                return self.get_current_units().get('moment', 'N*m')
            elif conj_dim == '[angle]':
                return 'rad'

            return base_unit

    def get_conjugate_unit_str(self, unit_str):
        """Get the work conjugate unit as a string"""
        return str(self.get_conjugate_unit(unit_str))

    def get_conjugate_units_array(self, units_array):
        """
        Convert an array of units to their work-conjugate units.

        Parameters
        ----------
        units_array : array-like
            Array of unit strings or pint.Unit objects

        Returns
        -------
        numpy.ndarray
            Array of conjugate units with the same shape as the input

        Examples
        --------
        >>> unit_manager.get_conjugate_units_array(['N', 'm', 'N*m'])
        array(['m', 'N', 'rad'], dtype=object)
        """
        import numpy as np

        # Convert input to numpy array if it's not already
        units_array = np.asarray(units_array, dtype=object)

        # Create output array with same shape
        result = np.empty_like(units_array, dtype=object)

        # Process each element
        for idx in np.ndindex(units_array.shape):
            unit = units_array[idx]
            result[idx] = self.get_conjugate_unit(unit)

        print(f"DEBUG: Converted units array to conjugates:\n  Original: {units_array}\n  Conjugates: {result}")

        return result








# Replace existing set_unit_system function to use the manager
def set_unit_system(system_dict, system_name=None):
    """Set the unit system throughout the package"""
    unit_manager.set_display_unit_system(system_dict, system_name)


def array_convert_to_unit_system(array, target_system_name, print_units=True):
    """
    Convert all pint.Quantity objects in a numpy array to units from a specified unit system.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing pint.Quantity objects.
    target_system_name : str
        Name of the unit system to convert to ('SI', 'imperial', or 'metric_kn').
    print_units : bool, optional
        If True, print the values with their units, default is True

    Returns
    -------
    numpy.ndarray
        Array with values converted to the target system units.
    """
    # Select the target unit system
    system_map = {
        'si': SI_UNITS,
        'imperial': IMPERIAL_UNITS,
        'metric_kn': METRIC_KN_UNITS,
        'metric': METRIC_KN_UNITS
    }
    target_system = system_map.get(target_system_name.lower(), unit_manager.get_current_units())

    # Simple mapping from dimensions to unit types
    dim_map = {
        FORCE_DIMENSIONALITY: 'force',
        LENGTH_DIMENSIONALITY: 'length',
        PRESSURE_DIMENSIONALITY: 'pressure',
        MOMENT_DIMENSIONALITY: 'moment',
        # ROTATION_DIMENSIONALITY: 'angle',
        DISTRIBUTED_LOAD_DIMENSIONALITY: 'distributed_load'
    }

    # Print header
    if print_units:
        print(f"\n=== Array Values in {target_system_name.upper()} Units ===")
        print(f"{'#':<3} {'Value':<12} {'Unit':<12} {'Type':<12}")
        print("-" * 40)

    # Process array
    result = []
    for i, val in enumerate(array):
        if isinstance(val, pint.Quantity):
            # Find unit type based on dimensionality
            unit_type = None
            for dim, utype in dim_map.items():
                if val.dimensionality == dim:
                    unit_type = utype
                    break

            if unit_type and unit_type in target_system:
                # Convert to target unit
                target_unit = target_system[unit_type]
                converted = val.to(target_unit)
                result.append(converted)

                if print_units:
                    print(f"{i:<3} {converted.magnitude:<12.4g} {target_unit:<12} {unit_type:<12}")
            else:
                # Cannot convert, keep original
                result.append(val)
                if print_units:
                    print(f"{i:<3} {val.magnitude:<12.4g} {val.units:<12} {'unknown':<12}")
        else:
            # Not a quantity
            result.append(val)
            if print_units:

                try:
                    print(f"{i:<3} {val:<12} {'N/A':<12} {'N/A':<12}")
                except TypeError as e:
                    print(f"i={i} val={val} (error: {e})")


    if print_units:
        print("-" * 40)

    return np.array(result, dtype=object)

def parse_value_with_units(value_str):
    """
    Parse a string with units using UnitManager's registry.

    Parameters
    ----------
    value_str : str
        String containing a value and potentially unit information.

    Returns
    -------
    pint.Quantity
        Parsed value with units.
    """
    try:
        return unit_manager.ureg.parse_expression(value_str)
    except Exception as e:
        print(f"Error parsing value '{value_str}': {e}")
        raise

def convert_to_unit(quantity, target_unit):
    """
    Convert a quantity to the target unit using UnitManager.

    Parameters
    ----------
    quantity : pint.Quantity
        Quantity to be converted.
    target_unit : str
        Target unit for conversion.

    Returns
    -------
    pint.Quantity
        Converted quantity.
    """
    try:
        return quantity.to(target_unit)
    except Exception as e:
        print(f"Error converting '{quantity}' to '{target_unit}': {e}")
        raise

# Example usage in a function
def process_node_data(node_data):
    """
    Process node data and ensure all values are in SI units.

    Parameters
    ----------
    node_data : dict
        Dictionary containing node information.

    Returns
    -------
    dict
        Processed node data with values in SI units.
    """
    x = parse_value_with_units(node_data["x"])
    y = parse_value_with_units(node_data["y"])

    x_meters = convert_to_unit(x, INTERNAL_LENGTH_UNIT).magnitude if isinstance(x, pint.Quantity) else x
    y_meters = convert_to_unit(y, INTERNAL_LENGTH_UNIT).magnitude if isinstance(y, pint.Quantity) else y

    return {"x": x_meters, "y": y_meters}


if __name__ == "__main__":
    # Example usage
    set_unit_system(IMPERIAL_UNITS, "imperial")
    print("Current unit system:", unit_manager.get_system_name())

    # Convert a value
    value = unit_manager.ureg.Quantity(10, 'N')
    converted_value = unit_manager.convert_value(value.magnitude, 'N', 'lbf')
    print(f"Converted {value} to {converted_value} lbf")

    # Convert a numpy array
    import numpy as np

    arr = np.array([unit_manager.ureg.Quantity(10, 'N'), unit_manager.ureg.Quantity(20, 'm')])
    converted_arr = array_convert_to_unit_system(arr, 'imperial')
    print("Converted array:", converted_arr)

    # Example usage
    array_with_units = np.array([
        unit_manager.ureg.Quantity(5, 'meter'),
        unit_manager.ureg.Quantity(10, 'newton')
    ], dtype=object)

    # Convert to imperial units
    imperial_values = array_convert_to_unit_system(array_with_units, 'imperial')
    print(imperial_values)  # Values in inches and klbf
