"""
Utilities for displaying structural engineering values with appropriate units.
"""
import numpy as np
# from pyMAOS.units_mod import ureg  # Replace this import
import pyMAOS
# Replace any direct ureg usage with unit_manager.ureg
Q_ = pyMAOS.unit_manager.ureg.Quantity

def get_unit_registry():
    """Get or create a unit registry for conversions"""
    try:
        import pyMAOS
        return pyMAOS.unit_manager.ureg
    except ImportError:
        from pint import UnitRegistry
        return UnitRegistry()

def display_node_load_vector_in_units(load_vector, node_uid, force_unit=None, length_unit=None, 
                                     load_combo_name=None, units_system=None):
    """
    Display a nodal load vector with appropriate units.
    
    Parameters
    ----------
    load_vector : list or ndarray
        Vector of [Fx, Fy, Mz]
    node_uid : int
        Node ID
    force_unit : str, optional
        Force unit (takes precedence over units_system)
    length_unit : str, optional
        Length unit (takes precedence over units_system)
    load_combo_name : str, optional
        Name of load combination
    units_system : dict, optional
        Dictionary containing unit definitions (e.g., SI_UNITS)
    """
    # First try specific units, then fall back to units_system if available
    force_unit = force_unit or (units_system.get("force") if units_system else pyMAOS.unit_manager.INTERNAL_FORCE_UNIT)
    length_unit = length_unit or (units_system.get("length") if units_system else pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT)

    moment_unit = f"{force_unit}*{length_unit}"
    
    # Get unit registry
    #
    
    # Convert load vector to display units
    fx_display = load_vector[0]
    fy_display = load_vector[1]
    mz_display = load_vector[2]
    
    try:
        # Try to convert with pint if needed
        fx_display = pyMAOS.unit_manager.ureg.Quantity(fx_display, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT).to(force_unit).magnitude
        fy_display = pyMAOS.unit_manager.ureg.Quantity(fy_display, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT).to(force_unit).magnitude
        mz_display = pyMAOS.unit_manager.ureg.Quantity(mz_display, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT).to(moment_unit).magnitude
    except:
        pass
    
    # Create load case string
    load_case_str = f" ({load_combo_name})" if load_combo_name else ""
    
    print(f"Node {node_uid} load{load_case_str}: "
          f"Fx={fx_display:.4g} {force_unit}, "
          f"Fy={fy_display:.4g} {force_unit}, "
          f"Mz={mz_display:.4g} {moment_unit}")

def display_node_displacement_in_units(displacement, node_uid, length_unit=None,
                                     load_combo_name=None, units_system=None):
    """
    Display nodal displacements with appropriate units.
    
    Parameters
    ----------
    displacement : list or ndarray
        Vector of [ux, uy, rz]
    node_uid : int
        Node ID
    length_unit : str, optional
        Length unit (takes precedence over units_system)
    load_combo_name : str, optional
        Name of load combination
    units_system : dict, optional
        Dictionary containing unit definitions (e.g., SI_UNITS)
    """
    # First try specific units, then fall back to units_system if available
    length_unit = length_unit or (units_system.get("length") if units_system else "m")
    
    # Get unit registry
    ureg = get_unit_registry()
    
    # Convert displacement vector to display units
    ux_display = displacement[0]
    uy_display = displacement[1]
    rz_display = displacement[2]  # Rotation remains in radians
    
    try:
        # Try to convert with pint if needed
        ux_display = pyMAOS.unit_manager.ureg.Quantity(ux_display, pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT).to(length_unit).magnitude
        uy_display = pyMAOS.unit_manager.ureg.Quantity(uy_display, pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT).to(length_unit).magnitude
    except:
        pass
    
    # Create load case string
    load_case_str = f" ({load_combo_name})" if load_combo_name else ""
    
    print(f"Node {node_uid} displacement{load_case_str}: "
          f"ux={ux_display:.6g} {length_unit}, "
          f"uy={uy_display:.6g} {length_unit}, "
          f"rz={rz_display:.6g} rad")

def display_member_forces_in_units(forces, member_uid, force_unit=None, length_unit=None,
                                 load_combo_name=None, units_system=None, location=None):
    """
    Display member end forces with appropriate units.
    
    Parameters
    ----------
    forces : list or ndarray
        Vector of [Fx1, Fy1, Mz1, Fx2, Fy2, Mz2]
    member_uid : int
        Member ID
    force_unit : str, optional
        Force unit (takes precedence over units_system)
    length_unit : str, optional
        Length unit (takes precedence over units_system)
    load_combo_name : str, optional
        Name of load combination
    units_system : dict, optional
        Dictionary containing unit definitions (e.g., SI_UNITS)
    location : str, optional
        Location description (e.g., "at x=2.5m")
    """
    # First try specific units, then fall back to units_system if available
    force_unit = force_unit or (units_system.get("force") if units_system else "N")
    length_unit = length_unit or (units_system.get("length") if units_system else "m")
    
    moment_unit = f"{force_unit}*{length_unit}"
    
    from pyMAOS.pymaos_units import ureg
    
    # Convert force values to display units
    display_forces = []
    for i, f in enumerate(forces):
        if i % 3 == 2:  # Every 3rd value is a moment
            try:
                display_forces.append(pyMAOS.unit_manager.ureg.Quantity(f, "N*m").to(moment_unit).magnitude)
            except:
                display_forces.append(f)
        else:  # Other values are forces
            try:
                display_forces.append(pyMAOS.unit_manager.ureg.Quantity(f, "N").to(force_unit).magnitude)
            except:
                display_forces.append(f)
    
    # Create location string
    location_str = f" {location}" if location else ""
    
    # Create load case string
    load_case_str = f" ({load_combo_name})" if load_combo_name else ""
    
    print(f"Member {member_uid}{location_str} forces{load_case_str}:")
    
    # For standard beam/frame with 6 components
    if len(display_forces) >= 6:
        print(f"  i-node: Fx={display_forces[0]:12.4g} {force_unit}, "
              f"Fy={display_forces[1]:12.4g} {force_unit}, "
              f"Mz={display_forces[2]:12.4g} {moment_unit}")
        print(f"  j-node: Fx={display_forces[3]:12.4g} {force_unit}, "
              f"Fy={display_forces[4]:12.4g} {force_unit}, "
              f"Mz={display_forces[5]:12.4g} {moment_unit}")
    else:
        # For other element types, just print all components
        components = [f"{v:12.4g}" for v in display_forces]
        print(f"  Forces: {', '.join(components)}")


import pint
from typing import List, Any, Union


def print_quantity_nested_list(data, indent=0, precision=4, width=15, simplify_units=True):
    """
    Print nested lists containing pint.Quantity objects with simplified units.

    Parameters
    ----------
    data : Any
        The data structure to print
    indent : int
        Current indentation level
    precision : int
        Decimal places to display
    width : int
        Minimum field width
    simplify_units : bool
        Whether to simplify units using to_reduced_units()
    """
    if isinstance(data, pint.Quantity):
        # Simplify units if requested
        if simplify_units:
            data = data.to_reduced_units()
        formatted = f"{data.magnitude:.{precision}g} {data.units}"
        print(formatted.ljust(width), end="")
    elif isinstance(data, list):
        # Handle lists (similar to your original implementation)
        if not data:
            print("[]")
            return

        all_simple = all(not isinstance(x, list) for x in data)
        if all_simple and len(data) <= 4:
            print("[", end="")
            for i, item in enumerate(data):
                print_quantity_nested_list(item, indent, precision, width=0, simplify_units=simplify_units)
                if i < len(data) - 1:
                    print(", ", end="")
            print("]")
        else:
            print("[")
            for i, item in enumerate(data):
                print(" " * (indent + 2), end="")
                print_quantity_nested_list(item, indent + 2, precision, width, simplify_units=simplify_units)
                if i < len(data) - 1:
                    print(" " * indent + ",")
                else:
                    print("")
            print(" " * indent + "]", end="")
            if indent==0:
                print()
    else:
        # Handle non-quantity, non-list values
        formatted = f"{data:.{precision}g}" if isinstance(data, (int, float)) else str(data)
        print(formatted.ljust(width), end="")

def print_quantity_list(data: List[Union[pint.Quantity, Any]], precision=4, width=15):
    """
    Print a list of pint.Quantity objects or other values in a formatted manner.

    Parameters
    ----------
    data : List[Union[pint.Quantity, Any]]
        List containing pint.Quantity objects or other values
    precision : int
        Number of decimal places to display for numerical values
    width : int
        Minimum width for each value field
    """
    print("[", end="")
    for i, item in enumerate(data):
        print_quantity_nested_list(item, indent=0, precision=precision, width=width)
        if i < len(data) - 1:
            print(", ", end="")
    print("]")  # Close the list

def print_quantity(data: Union[pint.Quantity, List[Union[pint.Quantity, Any]]],
                   precision=4, width=15):
    """
    Print a pint.Quantity or a list of pint.Quantity objects in a formatted manner.

    Parameters
    ----------
    data : Union[pint.Quantity, List[Union[pint.Quantity, Any]]]
        The data to print (can be a single Quantity or a list)
    precision : int
        Number of decimal places to display for numerical values
    width : int
        Minimum width for each value field
    """
    if isinstance(data, list):
        print_quantity_list(data, precision, width)
    else:
        print_quantity_nested_list(data, indent=0, precision=precision, width=width)
    print()  # Newline at the end


# Using dir() with string formatting to include methods
def dump_object(obj):
    attrs = dir(obj)
    result = []
    for attr in attrs:
        if not attr.startswith('__'):  # Skip magic methods
            try:
                value = getattr(obj, attr)
                result.append(f"{attr}: {value}")
            except Exception as e:
                result.append(f"{attr}: <Error: {e}>")
    return '\n'.join(result)

# For a more powerful solution, you can use the inspect module:


def object_to_string(obj):
    import inspect
    attributes = inspect.getmembers(obj, lambda a: not inspect.isroutine(a))
    attributes = [a for a in attributes if not a[0].startswith('__')]
    return '\n'.join(f"{attr}: {repr(val)}" for attr, val in attributes)

    def print_quantity_nested_list(nested_list, indent=0, indent_step=2):
        """
        Print a nested list of quantities using str() method for each Quantity object.

        Parameters:
        -----------
        nested_list : list
            The nested list containing Quantity objects
        indent : int
            Current indentation level
        indent_step : int
            Number of spaces for each indentation level
        """
        if isinstance(nested_list, list):
            print(" " * indent + "[")
            for item in nested_list:
                print_quantity_nested_list(item, indent + indent_step, indent_step)
            print(" " * indent + "]")
        else:
            # For quantity objects or other values
            print(" " * indent + str(nested_list) + ",")

    # Usage example
    print("Vy:")
    print_quantity_nested_list(Vy)


if __name__ == "__main__":
    # Example usage
    ureg = get_unit_registry()

    # Create some example quantities
    q1 = ureg.Quantity(5.123456, "m")
    q2 = ureg.Quantity(10.987654, "N")
    q3 = ureg.Quantity(3.14159, "rad")

    # Print a single quantity
    print_quantity(q1, precision=2)

    # Print a list of quantities
    print_quantity([q1, q2, q3], precision=2)

    # Print nested lists with quantities
    nested_data = [[q1, [q2, q3]], [q3, q1]]
    print_quantity(nested_data, precision=2)

    # Example for using the custom formatter
    def example():
        import pint
        ureg = pint.UnitRegistry()

        # Example similar to Dy structure in your code
        Dy = [
            [
                [0 * ureg.meter, 5.3 * ureg.newton * ureg.meter, 1.2 * ureg.newton],
                [0 * ureg.meter, 3.6 * ureg.meter]
            ],
            [
                [2.1 * ureg.meter, 7.8 * ureg.newton * ureg.meter, 3.4 * ureg.newton],
                [3.6 * ureg.meter, 5.0 * ureg.meter]
            ]
        ]

        print("Formatted output:")
        print_quantity_nested_list(Dy)

        # You can adjust precision and width
        print("\nWith custom precision and width:")
        print_quantity_nested_list(Dy, precision=2, width=20)
    example()

