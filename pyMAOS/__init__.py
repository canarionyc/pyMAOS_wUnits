"""
pyMAOS - Python Mechanics and Analysis of Structures

This package provides tools for structural engineering analysis,
with a focus on frame structures.
"""

# Import and expose unit systems and other global constants
from pyMAOS.units_mod import (SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
    INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT
)
from pprint import pprint
import pint

# Add references to the new modules
from .structure2d import R2Structure
from .structure2d_export import export_results_to_excel
from .structure2d_viz import plot_loadcombos_vtk
from .structure2d_utils import __str__

print("pyMAOS package initialized with unit systems and structural analysis tools.")

__all__ = [
    'SI_UNITS', 'IMPERIAL_UNITS', 'METRIC_KN_UNITS',
    'INTERNAL_FORCE_UNIT', 'INTERNAL_LENGTH_UNIT', 'INTERNAL_MOMENT_UNIT', 'INTERNAL_PRESSURE_UNIT', 'INTERNAL_DISTRIBUTED_LOAD_UNIT',
    'R2Structure'  # The module itself
]
# Package version
__version__ = '0.1.0'