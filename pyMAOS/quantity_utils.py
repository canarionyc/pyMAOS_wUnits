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

def increment_with_units(self, addend):
    """
    Increment a value with another value while ensuring consistent units.

    If self is not a Quantity, it's promoted to a Quantity with internal units.
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

    print(f"DEBUG: Incrementing {self} with {addend}")

    # If addend is not a Quantity, just do regular addition
    if not isinstance(addend, pint.Quantity):
        result = self + addend
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
        print(f"DEBUG: Using internal unit {internal_unit} for {unit_type}")
    else:
        # If we can't determine the unit type, use addend's units as fallback
        internal_unit = addend.units
        print(f"DEBUG: Could not determine unit type, using addend units {internal_unit}")

    # If self is not a Quantity, promote it to a Quantity with internal units
    if not isinstance(self, pint.Quantity):
        self = pyMAOS.unit_manager.ureg.Quantity(self, internal_unit)
        print(f"DEBUG: Promoted self to Quantity with internal units: {self}")

    try:
        # Convert addend to internal units before adding
        converted_addend = addend.to(internal_unit)
        print(f"DEBUG: Converted addend from {addend} to {converted_addend}")

        # Create result with the proper internal units
        result = type(self)(self.magnitude + converted_addend.magnitude, internal_unit)
        print(f"DEBUG: Result after increment: {result}")

        return result
    except pint.DimensionalityError as e:
        print(f"DEBUG: Dimensionality error - {self.dimensionality} ≠ {addend.dimensionality}")
        raise e

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

def print_units_matrix(array):
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