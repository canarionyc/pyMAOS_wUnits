"""
Utilities for handling loads with proper unit conversion.
"""
from typing import Union, Tuple, Dict, Any
import numpy as np

# Import unit systems from globals
from pyMAOS.units_mod import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

# Import the unit registry
try:
    from pyMAOS.units_mod import ureg, Q_
except ImportError:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    Q_ = ureg.Quantity

# def convert_load_value(value: Union[str, float, Q_],
#                        load_type: str,
#                        units_system: Dict[str, str]) -> Tuple[float, float, str, str]:
#     """
#     Convert load value to internal units and display units.
#
#     Parameters
#     ----------
#     value : str, float, or Quantity
#         The load value with or without units
#     load_type : str
#         Type of load (point_load, distributed_load, moment, temperature, etc.)
#     units_system : dict
#         Dictionary of unit definitions
#
#     Returns
#     -------
#     tuple
#         (internal_value, display_value, internal_unit, display_unit)
#     """
#     # Get appropriate units based on load type
#     internal_unit, display_unit = get_load_units(load_type, units_system)
#
#     # Parse the value if it's a string
#     if isinstance(value, str):
#         try:
#             # Try parsing as a Quantity with units
#             value = Q_(value)
#         except:
#             # If parsing fails, treat as a dimensionless number
#             value = float(value)
#
#     # Handle Quantity objects (with units)
#     if isinstance(value, Q_):
#         try:
#             # Convert to internal units
#             internal_value = value.to(internal_unit).magnitude
#             # Convert to display units
#             display_value = value.to(display_unit).magnitude
#             return internal_value, display_value, internal_unit, display_unit
#         except Exception as e:
#             print(f"Warning: Unit conversion error ({e}). Using magnitude directly.")
#             return value.magnitude, value.magnitude, internal_unit, display_unit
#
#     # Handle numeric values (without units)
#     return value, value, internal_unit, display_unit

def get_load_units(load_type: str, units_system: Dict[str, str]) -> Tuple[str, str]:
    """
    Get internal and display units for a specific load type.
    
    Parameters
    ----------
    load_type : str
        Type of load (point_load, distributed_load, moment, etc.)
    units_system : dict
        Dictionary of unit definitions
        
    Returns
    -------
    tuple
        (internal_unit, display_unit)
    """
    force_unit = units_system.get('force', 'N')
    length_unit = units_system.get('length', 'm')
    
    if load_type == 'point_load':
        return 'N', force_unit
    elif load_type == 'distributed_load':
        return 'N/m', f"{force_unit}/{length_unit}"
    elif load_type == 'moment':
        return 'N*m', f"{force_unit}*{length_unit}"
    elif load_type == 'temperature':
        return 'delta_degC', 'delta_degC'  # Temperature difference
    elif load_type == 'pressure':
        return 'Pa', units_system.get('pressure', 'Pa')
    elif load_type == 'strain':
        return 'dimensionless', 'dimensionless'
    else:
        # Default to force units
        return 'N', force_unit

class LoadConverter:
    """Class to handle load conversions with specified unit system"""
    
    def __init__(self, units_system=None):
        """
        Initialize with unit system
        
        Parameters
        ----------
        units_system : dict
            Dictionary of unit definitions
        """
        self.units_system = units_system or {
            'force': 'N',
            'length': 'm',
            'pressure': 'Pa'
        }
        
    def point_load(self, value, position=None, direction='Y'):
        """
        Convert a point load with optional position
        
        Parameters
        ----------
        value : str, float, or Quantity
            The load value
        position : float or str, optional
            Position along member
        direction : str, optional
            Direction of load (X, Y, Z)
            
        Returns
        -------
        dict
            Converted values with units
        """
        internal_value, display_value, internal_unit, display_unit = convert_load_value(
            value, 'point_load', self.units_system)
            
        result = {
            'internal': {
                'value': internal_value,
                'unit': internal_unit
            },
            'display': {
                'value': display_value,
                'unit': display_unit
            },
            'direction': direction
        }
        
        # Handle position if provided
        if position is not None:
            pos_internal, pos_display, pos_int_unit, pos_disp_unit = convert_load_value(
                position, 'length', self.units_system)
            
            result['position'] = {
                'internal': {
                    'value': pos_internal,
                    'unit': pos_int_unit
                },
                'display': {
                    'value': pos_display,
                    'unit': pos_disp_unit
                }
            }
            
        return result
        
    def distributed_load(self, w1_value, w2_value=None, a=0.0, b=None, direction='Y'):
        """
        Convert a distributed load
        
        Parameters
        ----------
        w1_value : str, float, or Quantity
            Start intensity value
        w2_value : str, float, or Quantity, optional
            End intensity value (defaults to w1 for uniform load)
        a : float or str, optional
            Start position
        b : float or str, optional
            End position (defaults to member length)
        direction : str, optional
            Direction of load
            
        Returns
        -------
        dict
            Converted values with units
        """
        # Process w1
        w1_internal, w1_display, w1_int_unit, w1_disp_unit = convert_load_value(
            w1_value, 'distributed_load', self.units_system)
            
        # Process w2 (defaults to w1 if not provided)
        if w2_value is None:
            w2_value = w1_value
            
        w2_internal, w2_display, w2_int_unit, w2_disp_unit = convert_load_value(
            w2_value, 'distributed_load', self.units_system)
            
        # Process position a
        a_internal, a_display, a_int_unit, a_disp_unit = convert_load_value(
            a, 'length', self.units_system)
            
        result = {
            'w1': {
                'internal': {'value': w1_internal, 'unit': w1_int_unit},
                'display': {'value': w1_display, 'unit': w1_disp_unit}
            },
            'w2': {
                'internal': {'value': w2_internal, 'unit': w2_int_unit},
                'display': {'value': w2_display, 'unit': w2_disp_unit}
            },
            'a': {
                'internal': {'value': a_internal, 'unit': a_int_unit},
                'display': {'value': a_display, 'unit': a_disp_unit}
            },
            'direction': direction
        }
        
        # Process position b if provided
        if b is not None:
            b_internal, b_display, b_int_unit, b_disp_unit = convert_load_value(
                b, 'length', self.units_system)
                
            result['b'] = {
                'internal': {'value': b_internal, 'unit': b_int_unit},
                'display': {'value': b_display, 'unit': b_disp_unit}
            }
            
        return result

    def moment(self, value, position=None):
        """
        Convert a moment load
        
        Parameters
        ----------
        value : str, float, or Quantity
            The moment value
        position : float or str, optional
            Position along member
            
        Returns
        -------
        dict
            Converted values with units
        """
        internal_value, display_value, internal_unit, display_unit = convert_load_value(
            value, 'moment', self.units_system)
            
        result = {
            'internal': {
                'value': internal_value,
                'unit': internal_unit
            },
            'display': {
                'value': display_value,
                'unit': display_unit
            }
        }
        
        # Handle position if provided
        if position is not None:
            pos_internal, pos_display, pos_int_unit, pos_disp_unit = convert_load_value(
                position, 'length', self.units_system)
            
            result['position'] = {
                'internal': {
                    'value': pos_internal,
                    'unit': pos_int_unit
                },
                'display': {
                    'value': pos_display,
                    'unit': pos_disp_unit
                }
            }
            
        return result
        
    def temperature_load(self, delta_t, alpha=1.2e-5):
        """
        Convert a temperature load
        
        Parameters
        ----------
        delta_t : float
            Temperature change
        alpha : float
            Coefficient of thermal expansion
            
        Returns
        -------
        dict
            Converted values
        """
        return {
            'delta_t': float(delta_t),
            'alpha': float(alpha),
            'strain': float(delta_t) * float(alpha)
        }
        
    def axial_load(self, value):
        """
        Convert an axial load
        
        Parameters
        ----------
        value : str, float, or Quantity
            The axial load value
            
        Returns
        -------
        dict
            Converted values with units
        """
        internal_value, display_value, internal_unit, display_unit = convert_load_value(
            value, 'point_load', self.units_system)
            
        return {
            'internal': {
                'value': internal_value,
                'unit': internal_unit
            },
            'display': {
                'value': display_value,
                'unit': display_unit
            }
        }
