# -*- coding: utf-8 -*-
import numpy as np
import pint
import pyMAOS

# Replace any direct ureg usage with unit_manager.ureg
Q_ = pyMAOS.unit_manager.ureg.Quantity

def convert_to_unit(array_with_units, target_unit):
    """Convert all quantities in the array to a common unit.

    Parameters
    ----------
    array_with_units : array of pint.Quantity
        Array containing quantities to convert
    target_unit : str or pint.Unit
        Target unit to convert all quantities to

    Returns
    -------
    numpy.ndarray
        Array with all values converted to the target unit
    """
    if isinstance(array_with_units, np.ndarray):
        # Convert to object array to handle mixed types
        result = np.empty_like(array_with_units, dtype=object)

        # Process each element
        for idx in np.ndindex(array_with_units.shape):
            value = array_with_units[idx]
            if isinstance(value, pint.Quantity):
                try:
                    # Try to convert to target unit
                    result[idx] = value.to(target_unit).magnitude
                except pint.DimensionalityError:
                    # If conversion not possible, keep original value
                    result[idx] = value
            else:
                result[idx] = value

        return result
    else:
        raise TypeError("Input must be a numpy array containing pint.Quantity objects")

# def convert_array_to_float64(input_array):
#     """
#     Convert an array with mixed types or Quantity objects to a uniform float64 array.
#
#     Parameters
#     ----------
#     input_array : array-like
#         Array that may contain Quantity objects or other numeric types
#
#     Returns
#     -------
#     numpy.ndarray
#         Converted array with dtype=float64
#     """
#     import numpy as np
#
#     # Convert to numpy array if not already
#     array = np.asarray(input_array)
#
#     # If already float64 array, return directly
#     if array.dtype == np.float64:
#         return array
#
#     # Create output array
#     result = np.empty(array.shape, dtype=np.float64)
#
#     # Process each element
#     for idx in np.ndindex(array.shape):
#         val = array[idx]
#
#         # Handle Quantity objects
#         if hasattr(val, 'magnitude'):
#             result[idx] = float(val.magnitude)
#         else:
#             # Handle other types
#             try:
#                 result[idx] = float(val)
#             except (TypeError, ValueError):
#                 print(f"DEBUG: Cannot convert value at {idx}: {val}")
#                 result[idx] = 0.0
#
#     # print(f"DEBUG: Converted array to float64: {result}")
#     return result

def numpy_array_of_quantity_to_numpy_array_of_float64(quantity_array):
    """
    Convert an array of mixed types or Quantity objects to a uniform float64 array.

    Parameters
    ----------
    quantity_array : array-like
        Array containing Pint Quantity objects or other numeric types

    Returns
    -------
    numpy.ndarray
        Array with the same shape as input but with magnitudes as float64 values
    """
    import numpy as np

    # Convert to numpy array if not already
    array = np.asarray(quantity_array)

    # Fast path if already float64 array
    if array.dtype == np.float64:
        return array

    # Fast path for arrays that are already numeric
    if np.issubdtype(array.dtype, np.number):
        print(f"DEBUG: Array already has numeric dtype {array.dtype}, converting to float64")
        return array.astype(np.float64)

    # For single Quantity objects
    if hasattr(array, 'magnitude'):
        print(f"DEBUG: Converting single Quantity with magnitude")
        return np.float64(array.magnitude)

    # Create output array
    result = np.zeros(array.shape, dtype=np.float64)

    # Use flat iteration for better performance
    for i, val in enumerate(array.flat):
        try:
            # Handle Quantity objects
            if hasattr(val, 'magnitude'):
                result.flat[i] = float(val.magnitude)
            else:
                # Handle other types
                result.flat[i] = float(val)
        except (TypeError, ValueError):
            idx = np.unravel_index(i, array.shape)
            print(f"DEBUG: Cannot convert value at {idx}: {val}")
            result.flat[i] = 0.0

    print(f"DEBUG: Converted array with shape {array.shape} to float64")
    return result

def extract_units_from_quantities(quantity_array):
    """
    Extract unit information from an array of Pint Quantity objects.

    Parameters
    ----------
    quantity_array : numpy.ndarray
        Array containing Pint Quantity objects

    Returns
    -------
    numpy.ndarray
        Array of the same shape containing only the unit information
    """
    import numpy as np

    # Create an empty array with object dtype to store unit objects
    units_array = np.empty_like(quantity_array, dtype=object)

    # Iterate through all elements
    for idx in np.ndindex(quantity_array.shape):
        # Get the quantity at this position
        quantity = quantity_array[idx]

        # Extract just the unit information if it's a Quantity
        if hasattr(quantity, 'units'):
            units_array[idx] = quantity.units
            print(f"DEBUG: Found unit {quantity.units} at position {idx}")
        else:
            units_array[idx] = None
            print(f"DEBUG: No unit at position {idx}")

    return units_array

# Example function: extract magnitude if it's a Quantity, else return as is
def to_float64(x):
	# Assume x is a Pint Quantity or a float
	return x.magnitude if hasattr(x, 'magnitude') else float(x)

# Vectorize the function
to_float64_vec = np.vectorize(to_float64)

to_float64_vec.__annotations__ = {'x': 'object', 'return': 'float64'}  # type: ignore

to_float64_ufunc = np.frompyfunc(to_float64, 1, 1)

def increment_with_units(target, addend):
    """
    Increment a value with another value while ensuring consistent units.

    If target is not a Quantity, it's promoted to a Quantity with internal units.
    If addend is a Quantity, it's converted to internal units before adding.

    Parameters
    ----------
    addend : pint.Quantity or scalar
        The value to add

    Returns
    -------
    pint.Quantity
        A new Quantity with internal units, incremented by addend
    """
    import pint

    # print(f"DEBUG: Incrementing {target} with {addend}")

    # If addend is not a Quantity, just do regular addition
    if not isinstance(addend, pint.Quantity):
        result = target + addend
        print(f"DEBUG: Added non-Quantity addend, result = {result}")
        return result

    # Determine the internal unit type based on addend's dimensionality
    unit_type = None
    if addend.check('[length]'):
        unit_type = 'length'
    elif addend.check('[force]'):
        unit_type = 'force'
    elif addend.check('[length] * [force]'):
        unit_type = 'moment'
    elif addend.check('[force] / [length]'):
        unit_type = 'distributed_load'
    elif addend.check('[force] / [length]^2'):
        unit_type = 'pressure'

    # Get the appropriate internal unit
    if unit_type:
        internal_unit = pyMAOS.unit_manager.get_internal_unit(unit_type)
        # print(f"DEBUG: Using internal unit {internal_unit} for {unit_type}")
    else:
        # If we can't determine the unit type, use addend's units as fallback
        internal_unit = addend.units
        # print(f"DEBUG: Could not determine unit type, using addend units {internal_unit}")

    # If target is not a Quantity, promote it to a Quantity with internal units
    if not isinstance(target, pint.Quantity):
        target = pyMAOS.unit_manager.ureg.Quantity(target, internal_unit)
        # print(f"DEBUG: Promoted target to Quantity with internal units: {target}")

    try:
        # Convert addend to internal units before adding
        converted_addend = addend.to(internal_unit)
        # print(f"DEBUG: Converted addend from {addend} to {converted_addend}")

        # Create result with the proper internal units
        result = type(target)(target.magnitude + converted_addend.magnitude, internal_unit)
        # print(f"DEBUG: Result after increment: {result}")

        return result
    except pint.DimensionalityError as e:
        print(f"DEBUG: Dimensionality error - {target.dimensionality} ≠ {addend.dimensionality}")
        raise e

increment_with_units_vec = np.vectorize(increment_with_units, otypes=[object])

increment_with_units_ufunc = np.frompyfunc(increment_with_units, 2, 1)

def add_arrays_with_units(array1, array2):
    """
    Add two arrays element-wise while ensuring consistent units for each element.

    For each element pair:
    - Checks that dimensions agree, or one is a pure number and the other a Quantity
    - Sums the magnitudes elementwise
    - Converts the result to a Quantity in the internal unit system

    Parameters
    ----------
    array1 : array-like
        First array, may contain Quantity objects
    array2 : array-like
        Second array, may contain Quantity objects

    Returns
    -------
    numpy.ndarray
        Result array with proper internal units for each element
    """
    import numpy as np
    from pyMAOS.quantity_utils import increment_with_units

    # Convert inputs to numpy arrays if they're not already
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)

    # Check that shapes are compatible
    if array1.shape != array2.shape:
        raise ValueError(f"Arrays must have the same shape, got {array1.shape} and {array2.shape}")

    # Create output array with the same shape
    result = np.empty_like(array1, dtype=object)

    # Process each element using the existing increment_with_units function
    for idx in np.ndindex(array1.shape):
        result[idx] = increment_with_units(array1[idx], array2[idx])
        print(f"DEBUG: Element-wise addition at {idx}: {array1[idx]} + {array2[idx]} = {result[idx]}")

    return result

def print_array_with_units(array):
    """
    Print a matrix with its values and units.

    This function displays the content of a numpy array containing Pint Quantity objects,
    showing both the magnitude values and their corresponding units.

    For elements without units, only their values are shown.

    Parameters
    ----------
    array : numpy.ndarray
        The array to print, potentially containing Pint Quantity objects
    """
    import numpy as np

    # First, print the shape information
    print(f"Matrix shape: {array.shape}")

    # Helper function for formatting a single value
    def format_value(val):
        if val is None:
            return "None"
        elif hasattr(val, 'units'):
            # Format magnitude with appropriate precision
            if abs(val.magnitude) < 1e-10:
                return f"0 {val.units}"
            else:
                return f"{val.magnitude:.4g} {val.units}"
        else:
            # Format plain numbers with appropriate precision
            if isinstance(val, (int, float)) and abs(val) < 1e-10:
                return "0"
            return str(val)

    # Helper function for recursive printing of subarrays
    def print_array(arr, indent=""):
        if arr.ndim == 1:
            elements = [format_value(val) for val in arr]
            print(indent + "[" + ", ".join(elements) + "]")
        elif arr.ndim == 2:
            print(indent + "[")
            for row in arr:
                print_array(row, indent + "  ")
            print(indent + "]")
        else:
            print(indent + f"Array with {arr.ndim} dimensions:")
            for i, subarray in enumerate(arr):
                print(indent + f"Dimension {i}:")
                print_array(subarray, indent + "  ")

    # Collect unit information for reporting
    if array.size > 0:
        unique_units = set()
        has_units = False

        # Check for units in the array
        for idx in np.ndindex(array.shape):
            val = array[idx]
            if hasattr(val, 'units'):
                has_units = True
                unique_units.add(str(val.units))

        if has_units:
            print(f"DEBUG: Units found in matrix: {', '.join(sorted(unique_units))}")
        else:
            print("DEBUG: No units found in matrix")

    # For empty arrays
    if array.size == 0:
        print("[]")
        return

    # For scalar arrays
    if array.ndim == 0:
        val = array.item()
        print(format_value(val))
        return

    # Print the array
    print_array(array)

def convert_registry(quantity, target_registry=None):
    """
    Convert a quantity to use a different registry.

    Parameters
    ----------
    quantity : pint.Quantity
        The quantity to convert
    target_registry : pint.UnitRegistry, optional
        The target registry. If None, uses the global registry from unit_manager.

    Returns
    -------
    pint.Quantity
        A new quantity using the target registry with the same value and unit
    """
    import pyMAOS

    # Default to global registry
    if target_registry is None:
        target_registry = pyMAOS.unit_manager.ureg

    # If already using target registry, return the original
    if hasattr(quantity, '_REGISTRY') and quantity._REGISTRY is target_registry:
        return quantity

    # Handle non-quantity objects
    if not hasattr(quantity, 'magnitude') or not hasattr(quantity, 'units'):
        return quantity

    # Create new quantity with target registry
    try:
        return target_registry.Quantity(quantity.magnitude, str(quantity.units))
    except Exception as e:
        print(f"Error converting quantity {quantity} to target registry: {e}")
        return quantity

def convert_all_quantities(obj, target_registry=None, processed_objects=None):
    """
    Recursively convert all quantities in a complex object to use the target registry.

    Parameters
    ----------
    obj : object
        The object containing quantities to convert
    target_registry : pint.UnitRegistry, optional
        The target registry. If None, uses the global registry from unit_manager.
    processed_objects : dict, optional
        Dictionary to track already processed objects to avoid circular references

    Returns
    -------
    object
        A copy of the object with all quantities using the target registry
    """
    import pyMAOS
    import numpy as np

    # Initialize processed_objects if it's the first call
    if processed_objects is None:
        processed_objects = {}

    # If we've already processed this object, return the converted version
    obj_id = id(obj)
    if obj_id in processed_objects:
        return processed_objects[obj_id]

    # Default to global registry
    if target_registry is None:
        target_registry = pyMAOS.unit_manager.ureg

    # Handle quantities directly
    if hasattr(obj, '_REGISTRY') and hasattr(obj, 'magnitude'):
        result = convert_registry(obj, target_registry)
        processed_objects[obj_id] = result
        return result

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        # For arrays of quantities
        if obj.dtype == object:
            result = np.empty_like(obj)
            for i, value in enumerate(obj.flat):
                result.flat[i] = convert_all_quantities(value, target_registry, processed_objects)
            processed_objects[obj_id] = result
            return result
        processed_objects[obj_id] = obj
        return obj

    # Handle lists
    if isinstance(obj, list):
        result = [convert_all_quantities(item, target_registry, processed_objects) for item in obj]
        processed_objects[obj_id] = result
        return result

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {key: convert_all_quantities(value, target_registry, processed_objects) for key, value in obj.items()}
        processed_objects[obj_id] = result
        return result

    # Handle objects with __dict__ attribute (most custom classes)
    if hasattr(obj, '__dict__'):
        # Create a shallow copy to avoid modifying original
        import copy
        new_obj = copy.copy(obj)

        # Mark this object as processed BEFORE recursing to prevent infinite loops
        processed_objects[obj_id] = new_obj

        # Convert all attributes, skipping properties that have no setter
        for attr_name, attr_value in obj.__dict__.items():
            try:
                # Check if this attribute is a property with no setter
                cls = obj.__class__
                if hasattr(cls, attr_name) and isinstance(getattr(cls, attr_name), property):
                    prop = getattr(cls, attr_name)
                    if prop.fset is None:
                        # Skip read-only properties
                        continue

                # Set the attribute with its converted value
                setattr(new_obj, attr_name, convert_all_quantities(attr_value, target_registry, processed_objects))
            except AttributeError as e:
                # This happens when trying to set read-only properties
                print(f"Warning: Could not set attribute '{attr_name}' on {obj.__class__.__name__}: {e}")
                continue

        return new_obj

    # Return other objects unchanged
    processed_objects[obj_id] = obj
    return obj

def convert_to_quantity_array(quantity_list, target_units=None):
    """
    Convert a list/array of individual quantities to a single quantity-wrapped numpy array,
    after verifying all elements have compatible units.

    Parameters
    ----------
    quantity_list : list or array
        List of quantities to convert
    target_units : str or pint.Unit, optional
        Target units for conversion. If None, uses the units of the first element.

    Returns
    -------
    pint.Quantity
        A single quantity object wrapping a numpy array, if all units are compatible.
        Otherwise returns the original list.

    Examples
    --------
    >>> from pint import UnitRegistry
    >>> ureg = UnitRegistry()
    >>> q_list = [ureg.Quantity(1.0, 'meter'), ureg.Quantity(2.0, 'meter')]
    >>> q_array = convert_to_quantity_array(q_list)
    >>> print(q_array)
    [1. 2.] meter
    """
    import numpy as np

    # Handle empty lists
    if not quantity_list:
        return quantity_list

    # Check if the input is already a wrapped quantity array
    if hasattr(quantity_list, 'magnitude') and isinstance(quantity_list.magnitude, np.ndarray):
        return quantity_list

    # Get the first element with units
    first_element = None
    for item in quantity_list:
        if hasattr(item, 'units'):
            first_element = item
            break

    # If no elements have units, return original
    if first_element is None:
        return quantity_list

    # Get the target units
    if target_units is None:
        target_units = first_element.units

    # Check if all elements have compatible units
    try:
        # Get the registry from the first element
        ureg = first_element._REGISTRY

        # Extract magnitudes and convert to target units if needed
        magnitudes = []
        for item in quantity_list:
            if hasattr(item, 'units'):
                if item.dimensionality == first_element.dimensionality:
                    # Convert to target units if needed
                    magnitudes.append(item.to(target_units).magnitude)
                else:
                    print(f"Warning: Incompatible units in array: {item.units} vs {first_element.units}")
                    return quantity_list  # Return original if units are incompatible
            else:
                print(f"Warning: Non-quantity element found in array: {item}")
                return quantity_list

        # Create a quantity-wrapped array
        return ureg.Quantity(np.array(magnitudes), target_units)

    except Exception as e:
        print(f"Error converting to quantity array: {e}")
        return quantity_list  # Return original on error

def uniquify_quantities(quantities):
    seen = set()
    unique_quantities = []

    for q in quantities:
        # Create a hashable key from the magnitude and units
        key = (float(q.magnitude), str(q.units))
        if key not in seen:
            seen.add(key)
            unique_quantities.append(q)

    return unique_quantities

# Usage example
# unique_list = uniquify_quantities([zero_length, target.a, target.b, target.L])

def uniquify_same_unit_quantities(quantities):
    """
    Uniquify a list of quantities that are known to have the same units.

    Parameters
    ----------
    quantities : list
        List of Quantity objects with the same units

    Returns
    -------
    list
        List of unique Quantity objects
    """
    # Extract the common unit from the first element
    units = quantities[0].units

    # Use a set for unique magnitude values
    unique_magnitudes = set()
    unique_quantities = []

    for q in quantities:
        # Convert to float for consistent comparison
        mag = float(q.magnitude)

        if mag not in unique_magnitudes:
            unique_magnitudes.add(mag)
            # Keep the original quantity object
            unique_quantities.append(q)

    return unique_quantities

# Example usage
unique_list = uniquify_same_unit_quantities([
    Q_(0, 'foot'),
    Q_(0.0, 'foot'),
    Q_(30.0, 'foot'),
    Q_(30.0, 'foot')
])
# Result: [<Quantity(0, 'foot')>, <Quantity(30.0, 'foot')>]