import numpy as np
import pint

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

class QuantityArray(np.ndarray):
    """
    A numpy array subclass that automatically extracts magnitude from pint.Quantity objects
    during assignment operations.
    """
    def __new__(cls, input_array, dtype=None, order=None):
        obj = np.asarray(input_array, dtype=dtype, order=order).view(cls)
        return obj

    def __setitem__(self, key, value):
        if isinstance(value, pint.Quantity):
            super().__setitem__(key, value.magnitude)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, np.ndarray) and value.dtype == object:
            # If the item is an array of objects, return as is
            return value
        elif isinstance(value, pint.Quantity):
            # If it's a pint.Quantity, return the magnitude
            return value.magnitude
        else:
            return value

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Ensure that the dtype is preserved
        self._dtype = getattr(obj, '_dtype', None)

import numpy as np
import pint

# Create a unit registry
ureg = pint.UnitRegistry()

def convert_to_unit(array, target_unit):
    """
    Convert all pint.Quantity objects in a numpy array to a target unit.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing pint.Quantity objects.
    target_unit : str
        Unit to which all quantities should be converted.

    Returns
    -------
    numpy.ndarray
        Array with magnitudes in the target unit.
    """
    converted_array = np.empty_like(array, dtype=float)
    for index, value in np.ndenumerate(array):
        if isinstance(value, pint.Quantity):
            converted_array[index] = value.to(target_unit).magnitude
        else:
            converted_array[index] = value  # Assume it's already a number
    return converted_array



if __name__ == "__main__":
    # Example usage
    import pint

    # Create a unit registry
    ureg = pint.UnitRegistry()

    # Define some quantities
    AE_L = 1.0 * ureg.meter  # Example quantity in meters

    # Create an instance of QuantityArray
    k_with_units = np.zeros((6, 6), dtype=object).view(QuantityArray)

    # Assign quantities to the array
    k_with_units[0, 0] = AE_L
    k_with_units[3, 3] = AE_L

    # Print the array to see the magnitudes
    print(2*k_with_units)

    # Example usage
    array_with_units = np.array([
        ureg.Quantity(5, 'meter'),
        ureg.Quantity(10, 'centimeter'),
        ureg.Quantity(2, 'kilometer')
    ], dtype=object)

    # Convert all quantities to meters
    converted_array = convert_to_unit(array_with_units, 'meter')
    print(converted_array)

