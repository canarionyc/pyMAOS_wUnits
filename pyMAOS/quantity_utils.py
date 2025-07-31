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

    def __new__(cls, input_array, units_dict=None):
        # Create a new array from the input
        obj = np.asarray(input_array).view(cls)
        # Store units dictionary
        obj._units_dict = units_dict if units_dict is not None else {}
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
    def __sub__(self, other):
        """
        Subtraction operator that handles unit inheritance when self has no units but other does.

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

        # Case where self has no units but other is a numpy array with Quantity objects
        if (not hasattr(self, '_units_dict') or not self._units_dict):
            # Handle case where other is a single pint Quantity
            if hasattr(other, 'units'):
                print(f"DEBUG: Inheriting units from single Quantity in subtraction: {other.units}")
                result._units_dict = {0: str(other.units)}

            # Handle case where other is an array containing pint Quantities
            elif isinstance(other, np.ndarray) and other.size > 0:
                # Check first non-None element for Quantity
                for item in other.flat:
                    if item is not None and hasattr(item, 'units'):
                        print(f"DEBUG: Inheriting units from array of Quantities: {item.units}")
                        # Create units dictionary from array items
                        result._units_dict = {}
                        for i, val in enumerate(other.flat):
                            if val is not None and hasattr(val, 'units'):
                                result._units_dict[i] = str(val.units)
                        break
        else:
            # Normal case - copy units from self
            if hasattr(self, '_units_dict'):
                result._units_dict = self._units_dict.copy()

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

    def print_units_matrix(self):
        """
        Print a matrix with its values and units.
        Uses the _units_dict to display units alongside magnitudes.
        """
        try:
            # Only proceed if it's a 2D array
            if self.ndim != 2:
                print(f"Input is not a 2D array. Shape: {self.shape}")
                return

            rows, cols = self.shape
            print(f"Matrix {rows}x{cols}:")

            # Format and print each row
            for i in range(rows):
                row_str = "["
                for j in range(cols):
                    # Calculate flat index for looking up units
                    flat_idx = i * cols + j

                    # Get value directly from array (magnitude only)
                    value = super(QuantityArray, self).__getitem__((i, j))

                    # Get unit if available
                    unit_str = ""
                    if flat_idx in self._units_dict:
                        unit_str = f" {self._units_dict[flat_idx]}"

                    row_str += f" {value:+8.3g}{unit_str:25s}"
                    if j < cols - 1:
                        row_str += ","
                row_str += " ]"
                print(row_str)
        except Exception as e:
            print(f"Error printing matrix: {e}")

    def add_with_units(self, position, value):
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
        Array containing pint.Quantity objects

    Returns
    -------
    numpy.ndarray
        A NumPy array of float64 values
    """
    # Check if it's already a regular numpy array without units
    if isinstance(quantity_array, np.ndarray) and not any(isinstance(x, pint.Quantity) for x in quantity_array.flat if hasattr(x, '__class__')):
        return quantity_array.astype(np.float64)

    # Extract shape to preserve structure
    original_shape = quantity_array.shape

    # Create result array
    result = np.empty(original_shape, dtype=np.float64)

    # Convert each element
    for idx in np.ndindex(original_shape):
        element = quantity_array[idx]
        if isinstance(element, pint.Quantity):
            result[idx] = element.magnitude
        else:
            # Handle non-Quantity values (like zeros)
            result[idx] = float(element)

    return result

# Alternative vectorized implementation for homogeneous arrays
def quantity_array_to_float64_vectorized(quantity_array):
    """Vectorized version for homogeneous arrays of Quantities"""
    if all(isinstance(x, pint.Quantity) for x in quantity_array.flat):
        return np.vectorize(lambda x: x.magnitude)(quantity_array).astype(np.float64)
    else:
        # Fall back to element-wise approach for mixed arrays
        return quantity_array_to_float64(quantity_array)





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
