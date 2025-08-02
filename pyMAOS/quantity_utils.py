import numpy as np
import pint
from pyMAOS.units_mod import unit_manager

# Replace any direct ureg usage with unit_manager.ureg
Q_ = unit_manager.ureg.Quantity

# Performance Considerations
# There is a performance penalty compared to standard numpy arrays:
#
#
# Type checking overhead: Every __setitem__ and __getitem__ operation performs a type check against pint.Quantity
# Unit extraction: Extra processing to extract magnitude values
# Method overriding: Custom methods add function call overhead
# The penalty is typically negligible for small to medium matrices but could be significant for:
#
#
# Large matrices (thousands of elements)
# Performance-critical code with frequent matrix operations
# Tight loops accessing many elements individually
# Important Limitations
# Unit information is lost: Only magnitudes are stored, not units
# No dimensional checking: The class doesn't verify unit compatibility in operations
# No unit conversion: Values must be converted to consistent units before assignment
# For occasional use with matrices where unit handling is important at boundaries but not during computation, QuantityArray provides a good balance of convenience and functionality.

import numpy as np
import pint
from pyMAOS.units_mod import unit_manager

class QuantityArray(np.ndarray):
    """
    Custom NumPy array that can track units for its elements.
    Used to perform calculations with quantities while preserving unit information.
    """

    def __new__(cls, input_array):
        """
        Create a new QuantityArray, preserving units if present.

        Parameters
        ----------
        input_array : array_like
            Input array or list of values, possibly with units

        Returns
        -------
        QuantityArray
            New array with units preserved
        """
        print(f"DEBUG: Creating QuantityArray from: {input_array}")

        # Check if we're dealing with a list/array of quantities with units
        if isinstance(input_array, (list, np.ndarray)) and len(input_array) > 0:
            has_units = any(hasattr(item, 'units') for item in input_array)

            if has_units:
                print(f"DEBUG: Input contains units, extracting magnitudes")
                # Extract magnitudes first
                magnitudes = np.array([item.magnitude if hasattr(item, 'magnitude') else item
                                      for item in input_array])

                # Create new array from magnitudes
                obj = np.asarray(magnitudes).view(cls)

                # Create units dictionary
                obj._units_dict = {}
                for i, item in enumerate(input_array):
                    if hasattr(item, 'units'):
                        obj._units_dict[i] = str(item.units)
                        print(f"DEBUG: Item {i} has units: {item.units}")

                return obj

        # Fall back to standard numpy array creation if no units
        print(f"DEBUG: No units detected, using standard numpy array creation")
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._units_dict = getattr(obj, '_units_dict', {})

    def __str__(self):
        """Display array with units attached to values"""
        if self.ndim == 0:  # Scalar case
            return self._format_with_units(self[()])

        # For 1D arrays
        if self.ndim == 1:
            elements = [self._format_with_units(val) for val in self]
            return f"[{', '.join(elements)}]"

        # For multi-dimensional arrays, use numpy's formatting but with units
        # Get the string representation of the array without units
        base_str = np.array_str(self)

        # Replace numeric values with unit-attached versions
        # This is a simple approach - for complex arrays a more sophisticated
        # implementation might be needed
        if self._units_dict:
            return f"QuantityArray with units {self._units_dict}:\n{base_str}"
        return base_str

    def _format_with_units(self, value):
        """Format a single value with its unit if available, avoiding unit duplication"""
        # If value is already a Quantity object, just return its string representation
        if hasattr(value, 'units'):
            return str(value)

        # If value is an index into the array
        if isinstance(value, (int, np.integer)) and value < len(self):
            value = self[value]
            # If after index access it's a Quantity, return its string representation
            if hasattr(value, 'units'):
                return str(value)

        # Find the unit for this position or value in the units dictionary
        unit = None
        for unit_key, unit_value in self._units_dict.items():
            if unit_key == value or (isinstance(unit_key, int) and unit_key < len(self)):
                unit = unit_value
                break

        # Prevent adding units to values that might already have units in string form
        if unit and not str(value).endswith(unit):
            return f"{value} {unit}"

        return f"{value}"

    def with_units(self):
        """Return array with units attached for display"""
        return self  # The __str__ method will handle unit display

    def __setitem__(self, key, value):
        if isinstance(value, pint.Quantity):
            # Store magnitude in array
            super().__setitem__(key, value.magnitude)

            # Store unit for this position
            if isinstance(key, tuple):
                # Handle multi-dimensional indices (e.g., [0, 0])
                flat_idx = np.ravel_multi_index(key, self.shape)
                self._units_dict[flat_idx] = value.units
            elif np.isscalar(key):
                # Single element assignment with scalar index
                self._units_dict[key] = value.units
            else:
                # Slice or advanced indexing assignment
                try:
                    # Get the indices affected by this assignment
                    mask = np.ones(self.shape)[key]
                    indices = np.where(np.atleast_1d(mask) if np.isscalar(mask) else mask)

                    for idx in zip(*indices):
                        flat_idx = np.ravel_multi_index(idx, self.shape)
                        self._units_dict[flat_idx] = value.units
                except (TypeError, ValueError):
                    # Fallback for unusual key types
                    print(f"Warning: Unusual key type in QuantityArray.__setitem__: {type(key)}")
        else:
            # Regular assignment without units
            super().__setitem__(key, value)

            # Remove any units for these indices if they exist
            if isinstance(key, tuple):
                flat_idx = np.ravel_multi_index(key, self.shape)
                if flat_idx in self._units_dict:
                    del self._units_dict[flat_idx]
            elif np.isscalar(key) and key in self._units_dict:
                del self._units_dict[key]

    def __getitem__(self, key):
        # Get the value using the parent class's __getitem__
        value = super().__getitem__(key)

        # Handle slice operations that return a new array
        if isinstance(value, np.ndarray) and isinstance(value, QuantityArray):
            # If we have a slice and units dictionary exists
            if hasattr(self, '_units_dict') and self._units_dict:
                original_shape = self.shape
                new_shape = value.shape

                # Create a new units dictionary for the sliced array
                new_units_dict = {}

                # Convert key to slice objects to work with
                full_key = self._normalize_key_to_tuple(key, len(original_shape))

                # Process each unit in the original dictionary
                for flat_idx, unit in self._units_dict.items():
                    # Get original indices for this flat index
                    try:
                        orig_indices = np.unravel_index(flat_idx, original_shape)

                        # Check if these indices are within our slice
                        in_slice = True
                        new_indices = []

                        for i, (idx, sl) in enumerate(zip(orig_indices, full_key)):
                            if isinstance(sl, slice):
                                # Handle slice
                                start = sl.start if sl.start is not None else 0
                                stop = sl.stop if sl.stop is not None else original_shape[i]
                                step = sl.step if sl.step is not None else 1

                                # Check if index is in the slice
                                if start <= idx < stop and (idx - start) % step == 0:
                                    new_indices.append((idx - start) // step)
                                else:
                                    in_slice = False
                                    break
                            elif isinstance(sl, int):
                                # Handle integer index
                                if idx == sl:
                                    # This dimension is removed in the result
                                    pass
                                else:
                                    in_slice = False
                                    break
                            else:
                                # Other advanced indexing cases
                                # For simplicity, we don't handle these yet
                                in_slice = False
                                break

                        if in_slice and len(new_indices) == len(new_shape):
                            # Calculate new flat index in the sliced array
                            new_flat_idx = np.ravel_multi_index(tuple(new_indices), new_shape)
                            new_units_dict[new_flat_idx] = unit
                    except (ValueError, TypeError) as e:
                        # Skip indices that can't be unraveled or processed
                        print(f"DEBUG: Error processing index {flat_idx}: {e}")
                        continue

                # Set the new units dictionary on the sliced array
                value._units_dict = new_units_dict
                print(f"DEBUG: Updated units dict for slice: original had {len(self._units_dict)} entries, new has {len(new_units_dict)}")

            return value

        # If this is a scalar value, check if we have units for it
        if np.isscalar(value):
            # Calculate flat index for the key
            if isinstance(key, tuple):
                try:
                    flat_idx = np.ravel_multi_index(key, self.shape)
                    # If we have a unit for this index, return a Quantity
                    if flat_idx in self._units_dict:
                        from pyMAOS.units_mod import unit_manager
                        return unit_manager.ureg.Quantity(value, self._units_dict[flat_idx])
                except (ValueError, TypeError):
                    # Handle cases where ravel_multi_index might fail
                    pass
            elif np.isscalar(key) and key in self._units_dict:
                # Direct scalar indexing
                from pyMAOS.units_mod import unit_manager
                return unit_manager.ureg.Quantity(value, self._units_dict[key])

        # If no units or non-scalar result, return value as-is
        return value

    def _normalize_key_to_tuple(self, key, ndim):
        """Convert any key to a tuple of slices with the same length as ndim"""
        if not isinstance(key, tuple):
            # Single key becomes (key, slice(None), slice(None), ...)
            return (key,) + (slice(None),) * (ndim - 1)

        # If key is already a tuple but shorter than ndim
        if len(key) < ndim:
            return key + (slice(None),) * (ndim - len(key))

        return key

    def get_units(self):
        """
        Returns the units of the elements in the array.

        Returns
        -------
        pint.Unit or None
            The unit object for the array, or None if no units are found.
        """
        from pyMAOS.units_mod import unit_manager

        if not hasattr(self, '_units_dict') or not self._units_dict:
            print("DEBUG: No units dictionary found in QuantityArray")
            return None

        print(f"DEBUG: Units dictionary contains {len(self._units_dict)} entries")
        print(f"DEBUG: Keys in _units_dict: {list(self._units_dict.keys())}")

        # For arrays with a single unit (most common case)
        if len(self._units_dict) == 1:
            unit_str = next(iter(self._units_dict.values()))
            print(f"DEBUG: Found single unit string: {unit_str}")
            # Convert string to pint.Unit object
            return unit_manager.ureg.parse_units(unit_str)

        # For 1D arrays, try to find unit for first element
        if self.ndim == 1 and len(self) > 0:
            for idx, unit_str in self._units_dict.items():
                if idx == 0 or idx == (0,):
                    print(f"DEBUG: Using first element unit: {unit_str}")
                    return unit_manager.ureg.parse_units(unit_str)

        # Check if all units are the same
        units = list(self._units_dict.values())
        if len(units) > 0 and all(u == units[0] for u in units):
            print(f"DEBUG: All units are the same: {units[0]}")
            return unit_manager.ureg.parse_units(units[0])

        print("DEBUG: Multiple different units found - using first unit")
        first_unit = next(iter(self._units_dict.values()))
        return unit_manager.ureg.parse_units(first_unit)
    def __add__(self, other):
        """
        Addition operator that handles unit inheritance.
        When one operand has units and the other doesn't, the result inherits units from the one with units.

        Parameters
        ----------
        other : array_like or scalar
            Value to add to self

        Returns
        -------
        QuantityArray
            Result with appropriate units
        """
        # Create result using standard addition
        result = super().__add__(other)

        # Case 1: Self has units
        if hasattr(self, '_units_dict') and self._units_dict:
            print(f"DEBUG: Self has units, copying to result")
            result._units_dict = self._units_dict.copy()

        # Case 2: Self doesn't have units but other might have units
        else:
            # Check if other is a QuantityArray with units
            if isinstance(other, QuantityArray) and hasattr(other, '_units_dict') and other._units_dict:
                print(f"DEBUG: Inheriting units from QuantityArray")
                result._units_dict = other._units_dict.copy()

            # Check if other is a single Quantity with units
            elif hasattr(other, 'units'):
                print(f"DEBUG: Inheriting units from single Quantity in addition: {other.units}")
                result._units_dict = {0: str(other.units)}

            # Check if other is a numpy array containing Quantities
            elif isinstance(other, np.ndarray) and other.size > 0:
                # Process all elements in a single pass to find all units
                result._units_dict = {}
                has_units = False

                print(f"DEBUG: Checking numpy array for Quantities")

                # Iterate through all elements
                for i, item in enumerate(other.flat):
                    if item is not None and hasattr(item, 'units'):
                        result._units_dict[i] = str(item.units)
                        has_units = True
                        print(f"DEBUG: Found unit at position {i}: {item.units}")

                if has_units:
                    print(f"DEBUG: Inherited {len(result._units_dict)} units from array")
                else:
                    print(f"DEBUG: No units found in the array")

        return result
    def __sub__(self, other):
        """
        Subtraction operator that handles unit inheritance.
        When one operand has units and the other doesn't, the result inherits units from the one with units.

        Parameters
        ----------
        other : array_like or scalar
            Value to subtract from self

        Returns
        -------
        QuantityArray
            Result with appropriate units
        """
        # Create result using standard subtraction
        result = super().__sub__(other)

        # Case 1: Self has units
        if hasattr(self, '_units_dict') and self._units_dict:
            print(f"DEBUG: Self has units, copying to result")
            result._units_dict = self._units_dict.copy()

        # Case 2: Self doesn't have units but other might have units
        else:
            # Check if other is a QuantityArray with units
            if isinstance(other, QuantityArray) and hasattr(other, '_units_dict') and other._units_dict:
                print(f"DEBUG: Inheriting units from QuantityArray")
                result._units_dict = other._units_dict.copy()

            # Check if other is a single Quantity with units
            elif hasattr(other, 'units'):
                print(f"DEBUG: Inheriting units from single Quantity in subtraction: {other.units}")
                result._units_dict = {0: str(other.units)}

            # Check if other is a numpy array containing Quantities
            # Check if other is a numpy array containing Quantities
            elif isinstance(other, np.ndarray) and other.size > 0:
                # Process all elements in a single pass to find all units
                result._units_dict = {}
                has_units = False

                print(f"DEBUG: Checking numpy array for Quantities")

                # Iterate through all elements
                for i, item in enumerate(other.flat):
                    if item is not None and hasattr(item, 'units'):
                        result._units_dict[i] = str(item.units)
                        has_units = True
                        print(f"DEBUG: Found unit at position {i}: {item.units}")

                if has_units:
                    print(f"DEBUG: Inherited {len(result._units_dict)} units from array")
                else:
                    print(f"DEBUG: No units found in the array")
        return result

    def __mul__(self, other):
        """Handle multiplication while preserving units"""
        result = super().__mul__(other)
        # Units remain the same after scalar multiplication
        return result

    def __rmul__(self, other):
        """Handle right multiplication (scalar * array)"""
        result = super().__rmul__(other)
        # Units remain the same after scalar multiplication
        return result

    def view(self, dtype=None, type=None):
        """Ensure views preserve unit information"""
        result = super().view(dtype, type)
        if type is QuantityArray or isinstance(result, QuantityArray):
            result._units_dict = self._units_dict.copy()
        return result

    def copy(self):
        """Ensure copies preserve unit information"""
        result = super().copy()
        result._units_dict = self._units_dict.copy()
        return result

    def __repr__(self):
        """Representation showing it's a QuantityArray"""
        return f"QuantityArray({np.asarray(self).__repr__()})"



    def incremental_add_with_units(self, position, value):
        """
        Add a value to a specific position in the array while preserving units.
        Checks for unit consistency and only adds new units when needed.

        Parameters
        ----------
        position : tuple
            Position in the array (i, j, ...)
        value : scalar or pint.Quantity
            Value to add
        """
        # Calculate flat index for looking up units
        flat_idx = np.ravel_multi_index(position, self.shape)

        # Get current value
        current_value = self[position]

        print(f"DEBUG: Adding at position {position}: value type={type(value)}")
        print(f"DEBUG: Is Quantity? {isinstance(value, pint.Quantity)}")

        # Check if we're adding a Quantity with units
        if isinstance(value, pint.Quantity):
            value_magnitude = value.magnitude
            value_unit = value.units

            print(f"DEBUG: Found Quantity value={value}, magnitude={value_magnitude}, unit={value_unit}")

            # Check if position already has a unit assigned
            if flat_idx in self._units_dict:
                existing_unit = self._units_dict[flat_idx]
                print(f"DEBUG: Position already has unit {existing_unit}")

                try:
                    # Convert the incoming value to match the existing unit
                    converted_value = value.to(existing_unit)
                    print(f"DEBUG: Converted value to {existing_unit}: {converted_value}")
                    value_magnitude = converted_value.magnitude

                    # Add the magnitude to current value
                    new_value = current_value + value_magnitude
                except pint.DimensionalityError:
                    # If units can't be converted, create a new Quantity from current value
                    # with the existing unit, then add
                    current_quantity = unit_manager.ureg.Quantity(current_value, existing_unit)
                    result_quantity = current_quantity + value
                    new_value = result_quantity.magnitude

                    # Update the unit if necessary (e.g., if result has derived units)
                    self._units_dict[flat_idx] = result_quantity.units
                    print(f"DEBUG: Updated unit to {result_quantity.units} after addition")
            else:
                # No existing unit, just add magnitude and assign the unit
                new_value = current_value + value_magnitude
                self._units_dict[flat_idx] = value_unit
                print(f"DEBUG: Added unit {value_unit} at position {position}, flat_idx={flat_idx}")
        else:
            # No units on incoming value
            if flat_idx in self._units_dict:
                # If the position has a unit, treat the incoming value as a magnitude in that unit
                print(f"DEBUG: Adding unitless value to position with unit {self._units_dict[flat_idx]}")
                new_value = current_value + value
            else:
                # Both are unitless, just add
                new_value = current_value + value

        # Store the result
        super().__setitem__(position, new_value)
        print(f"DEBUG: Final value at position {position}: {new_value}")

# def _quantity_array_str(self):
#     return format_object_array(self)
# QuantityArray.__str__ = _quantity_array_str
#
def format_object_array(arr, precision=4):
    """
    Format a numpy array of objects (especially Pint quantities) using
    each element's __str__ representation.
    """
    if not isinstance(arr, np.ndarray):
        return str(arr)

    # Handle QuantityArray with stored units
    if isinstance(arr, QuantityArray) and hasattr(arr, '_units_array'):
        # Convert to array with reattached units first
        arr = arr.with_units()

    # For empty arrays
    if arr.size == 0:
        return "[]"

    # For scalar arrays
    if arr.ndim == 0:
        return str(arr.item())

    # For 1D arrays
    if arr.ndim == 1:
        elements = [str(x) for x in arr]
        return "[" + ", ".join(elements) + "]"

    # For 2D arrays
    if arr.ndim == 2:
        rows = []
        for row in arr:
            row_str = "[" + ", ".join(str(x) for x in row) + "]"
            rows.append(row_str)
        return "[" + ",\n ".join(rows) + "]"

    # For higher dimensions, fall back to numpy's repr
    return np.array2string(arr, formatter={'all': lambda x: str(x)})





import numpy as np
import pint
from pyMAOS.units_mod import unit_manager

# Replace any direct ureg usage with unit_manager.ureg
Q_ = unit_manager.ureg.Quantity

def convert_to_unit(array_with_units, target_unit):
    """Convert all quantities in the array to a common unit.

    Parameters
    ----------
    array_with_units : QuantityArray or array of pint.Quantity
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

# def format_object_array(arr, precision=4):
#     """
#     Format a numpy array of objects (especially Pint quantities) using
#     each element's __str__ representation.
#
#     Parameters
#     ----------
#     arr : numpy.ndarray
#         Array with dtype='object' containing elements with __str__ method
#     precision : int, optional
#         Number of decimal places for formatting numbers, by default 4
#
#     Returns
#     -------
#     str
#         Formatted string representation of the array
#     """
#     if not isinstance(arr, np.ndarray):
#         return str(arr)
#
#     # For empty arrays
#     if arr.size == 0:
#         return "[]"
#
#     # For scalar arrays
#     if arr.ndim == 0:
#         return str(arr.item())
#
#     # For 1D arrays
#     if arr.ndim == 1:
#         elements = [str(x) for x in arr]
#         return "[" + ", ".join(elements) + "]"
#
#     # For 2D arrays
#     if arr.ndim == 2:
#         rows = []
#         for row in arr:
#             row_str = "[" + ", ".join(str(x) for x in row) + "]"
#             rows.append(row_str)
#         return "[" + ",\n ".join(rows) + "]"
#
#     # For higher dimensions, fall back to numpy's repr
#     return np.array2string(arr, formatter={'all': lambda x: str(x)})
#
# # Register the function as a method for QuantityArray
# def _quantity_array_str(self):
#     return format_object_array(self)
#
# QuantityArray.__str__ = _quantity_array_str

import numpy as np
import pint

def quantity_array_to_float64(quantity_array):
    """
    Convert a QuantityArray or array containing pint.Quantity objects to
    a standard NumPy array of float64 values for use with linear algebra libraries.

    Parameters
    ----------
    quantity_array : QuantityArray or array-like
        Array containing Pint Quantity objects or numeric values

    Returns
    -------
    numpy.ndarray
        Array of float64 values with units stripped
    """
    import numpy as np

    # Fast path for arrays that are already numeric
    if np.issubdtype(quantity_array.dtype, np.number):
        print(f"DEBUG: Fast path - array already has numeric dtype {quantity_array.dtype}")
        return quantity_array.astype(np.float64)

    # For single Quantity objects
    if hasattr(quantity_array, 'magnitude'):
        print(f"DEBUG: Converting single Quantity with magnitude")
        return np.float64(quantity_array.magnitude)

    # Create output array directly using numpy
    result = np.zeros(quantity_array.shape, dtype=np.float64)

    # Use flat iteration for efficiency
    for i, val in enumerate(quantity_array.flat):
        # Extract magnitude if it's a Quantity, otherwise use the value directly
        result.flat[i] = val.magnitude if hasattr(val, 'magnitude') else float(val)

    print(f"DEBUG: Converted array with shape {quantity_array.shape} to float64")
    return result

# Alternative vectorized implementation for homogeneous arrays
def quantity_array_to_float64_vectorized(quantity_array):
    """Vectorized version for homogeneous arrays of Quantities"""
    if all(isinstance(x, pint.Quantity) for x in quantity_array.flat):
        return np.vectorize(lambda x: x.magnitude)(quantity_array).astype(np.float64)
    else:
        # Fall back to element-wise approach for mixed arrays
        return quantity_array_to_float64(quantity_array)

def convert_array_to_float64(input_array):
    """
    Convert an array with mixed types or Quantity objects to a uniform float64 array.

    Parameters
    ----------
    input_array : array-like
        Array that may contain Quantity objects or other numeric types

    Returns
    -------
    numpy.ndarray
        Converted array with dtype=float64
    """
    import numpy as np

    # Convert to numpy array if not already
    array = np.asarray(input_array)

    # If already float64 array, return directly
    if array.dtype == np.float64:
        return array

    # Create output array
    result = np.empty(array.shape, dtype=np.float64)

    # Process each element
    for idx in np.ndindex(array.shape):
        val = array[idx]

        # Handle Quantity objects
        if hasattr(val, 'magnitude'):
            result[idx] = float(val.magnitude)
        else:
            # Handle other types
            try:
                result[idx] = float(val)
            except (TypeError, ValueError):
                print(f"DEBUG: Cannot convert value at {idx}: {val}")
                result[idx] = 0.0

    print(f"DEBUG: Converted array to float64: {result}")
    return result

def quantity_array_to_numpy(quantity_array):
    """
    Convert a numpy array of Quantity objects to a numpy array of their
    magnitude values stored as float64.

    Parameters
    ----------
    quantity_array : numpy.ndarray
        Array containing Pint Quantity objects

    Returns
    -------
    numpy.ndarray
        Array with the same shape as input but with magnitudes as float64 values
    """
    import numpy as np

    # Fast path for arrays that are already numeric
    if np.issubdtype(quantity_array.dtype, np.number):
        print(f"DEBUG: Array already has numeric dtype {quantity_array.dtype}, converting to float64")
        return quantity_array.astype(np.float64)

    # For single Quantity objects
    if hasattr(quantity_array, 'magnitude'):
        print(f"DEBUG: Converting single Quantity with magnitude")
        return np.float64(quantity_array.magnitude)

    # Create output array with same shape but float64 dtype
    result = np.zeros(quantity_array.shape, dtype=np.float64)

    # Use flat iteration to extract magnitude values
    for i, val in enumerate(quantity_array.flat):
        # Extract magnitude if it's a Quantity, otherwise use value directly
        result.flat[i] = val.magnitude if hasattr(val, 'magnitude') else float(val)

    print(f"DEBUG: Converted array with shape {quantity_array.shape} to float64")
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

if __name__ == "__main__":
    # Example usage
    import pint
    AE_L = 1.0 * unit_manager.ureg.meter  # Example quantity in meters

    # Create an instance of QuantityArray
    k_with_units = np.zeros((6, 6), dtype=object).view(QuantityArray)

    # Assign quantities to the array
    k_with_units[0, 0] = AE_L
    k_with_units[3, 3] = AE_L

    # Print the array to see the magnitudes
    print(2*k_with_units)

    # Example usage
    array_with_units = np.array([
        unit_manager.ureg.Quantity(5, 'meter'),
        unit_manager.ureg.Quantity(10, 'centimeter'),
        unit_manager.ureg.Quantity(2, 'kilometer')
    ], dtype=object)

    # Convert all quantities to meters
    converted_array = convert_to_unit(array_with_units, 'meter')
    print(converted_array)# Create array with mixed units

    qarray=array_with_units.view(QuantityArray)
    print(qarray)
    print(qarray.with_units())  # Should print quantities with units reattached

    # Example usage of QuantityArray with new implementation
    from pyMAOS.quantity_utils import QuantityArray
    arr = QuantityArray([1, 2, 3])
    arr[0] = unit_manager.ureg.Quantity(10, 'newton')
    arr[1] = unit_manager.ureg.Quantity(20, 'meter')
    print(arr); arr_2=2*arr; print(arr_2)  # [20 newton, 40 meter, 6]
    # Get with units reattached
    quantities = arr.with_units()
    print(quantities)  # [10 newton, 20 meter, 3]

    # Example usage
    from scipy import linalg

    # Convert stiffness matrix to float64 array
    k_float = quantity_array_to_float64(stiffness_matrix)

    # Now you can use it with linear algebra functions
    inv_k = linalg.inv(k_float)
    solution = linalg.solve(k_float, force_vector_float)

    # Use it with your quantity array
    print_units_matrix(local_k)


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
    from pyMAOS.units_mod import unit_manager, get_internal_unit

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
        internal_unit = get_internal_unit(unit_type)
        print(f"DEBUG: Using internal unit {internal_unit} for {unit_type}")
    else:
        # If we can't determine the unit type, use addend's units as fallback
        internal_unit = addend.units
        print(f"DEBUG: Could not determine unit type, using addend units {internal_unit}")

    # If self is not a Quantity, promote it to a Quantity with internal units
    if not isinstance(self, pint.Quantity):
        self = unit_manager.ureg.Quantity(self, internal_unit)
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