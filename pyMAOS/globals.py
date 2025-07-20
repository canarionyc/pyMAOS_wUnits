"""
Global constants and configurations for the pyMAOS package.

This module provides access to commonly used constants and configurations,
particularly the predefined unit systems that are used throughout the package.
"""

from pyMAOS.units_mod import (
    SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
    DISPLAY_UNITS, FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, 
    PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT
)

# Re-export the predefined unit systems for easy access
__all__ = [
    'SI_UNITS', 'IMPERIAL_UNITS', 'METRIC_KN_UNITS', 
    'DISPLAY_UNITS', 'FORCE_UNIT', 'LENGTH_UNIT', 
    'MOMENT_UNIT', 'PRESSURE_UNIT', 'DISTRIBUTED_LOAD_UNIT'
]

# Default unit system to use if none specified
DEFAULT_UNIT_SYSTEM = METRIC_KN_UNITS

# Add any other global constants here
GRAVITY = 9.81  # m/s²

# Unit system definitions (for reference - actual data comes from units.py)
# SI_UNITS = {"force": "N", "length": "m", "pressure": "Pa", "distance": "m"}
# IMPERIAL_UNITS = {"force": "klbf", "length": "in", "pressure": "ksi", "distance": "ft"}
# METRIC_KN_UNITS = {"force": "kN", "length": "m", "pressure": "kN/m^2", "distance": "m"}
