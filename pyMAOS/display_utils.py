"""
Utilities for displaying structural engineering values with appropriate units.
"""
import numpy as np
from pint import UnitRegistry
import os

def get_unit_registry():
    """Get or create a unit registry for conversions"""
    try:
        from pyMAOS.units_mod import ureg
        return ureg
    except ImportError:
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
    force_unit = force_unit or (units_system.get("force") if units_system else "N")
    length_unit = length_unit or (units_system.get("length") if units_system else "m")
    
    moment_unit = f"{force_unit}*{length_unit}"
    
    # Get unit registry
    ureg = get_unit_registry()
    
    # Convert load vector to display units
    fx_display = load_vector[0]
    fy_display = load_vector[1]
    mz_display = load_vector[2]
    
    try:
        # Try to convert with pint if needed
        fx_display = ureg.Quantity(fx_display, "N").to(force_unit).magnitude
        fy_display = ureg.Quantity(fy_display, "N").to(force_unit).magnitude
        mz_display = ureg.Quantity(mz_display, "N*m").to(moment_unit).magnitude
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
        ux_display = ureg.Quantity(ux_display, "m").to(length_unit).magnitude
        uy_display = ureg.Quantity(uy_display, "m").to(length_unit).magnitude
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
    
    # Get unit registry
    ureg = get_unit_registry()
    
    # Convert force values to display units
    display_forces = []
    for i, f in enumerate(forces):
        if i % 3 == 2:  # Every 3rd value is a moment
            try:
                display_forces.append(ureg.Quantity(f, "N*m").to(moment_unit).magnitude)
            except:
                display_forces.append(f)
        else:  # Other values are forces
            try:
                display_forces.append(ureg.Quantity(f, "N").to(force_unit).magnitude)
            except:
                display_forces.append(f)
    
    # Create location string
    location_str = f" {location}" if location else ""
    
    # Create load case string
    load_case_str = f" ({load_combo_name})" if load_combo_name else ""
    
    print(f"Member {member_uid}{location_str} forces{load_case_str}:")
    
    # For standard beam/frame with 6 components
    if len(display_forces) >= 6:
        print(f"  i-node: Fx={display_forces[0]:.4g} {force_unit}, "
              f"Fy={display_forces[1]:.4g} {force_unit}, "
              f"Mz={display_forces[2]:.4g} {moment_unit}")
        print(f"  j-node: Fx={display_forces[3]:.4g} {force_unit}, "
              f"Fy={display_forces[4]:.4g} {force_unit}, "
              f"Mz={display_forces[5]:.4g} {moment_unit}")
    else:
        # For other element types, just print all components
        components = [f"{v:.4g}" for v in display_forces]
        print(f"  Forces: {', '.join(components)}")

def display_stiffness_matrix_in_units(matrix, units_system=None):
    """
    Display stiffness matrix with appropriate units notation.
    
    Parameters
    ----------
    matrix : ndarray
        Stiffness matrix
    units_system : dict, optional
        Dictionary containing unit definitions (e.g., SI_UNITS)
    """
    # Determine units based on units_system
    force_unit = units_system.get("force", "N") if units_system else "N"
    length_unit = units_system.get("length", "m") if units_system else "m"
    
    # Create unit descriptions for different terms in the stiffness matrix
    force_disp = f"{force_unit}/{length_unit}"      # Force/displacement
    force_rot = f"{force_unit}"                     # Force/rotation
    moment_disp = f"{force_unit}*{length_unit}/{length_unit}" # Moment/displacement = Force
    moment_rot = f"{force_unit}*{length_unit}"      # Moment/rotation
    
    # Print matrix with units note
    print(f"Stiffness Matrix (Various units: {force_disp}, {moment_rot}, etc.)")
    np.set_printoptions(precision=4, suppress=True)
    print(matrix)