"""
pyMAOS - Python Mechanics and Analysis of Structures

This package provides tools for structural engineering analysis,
with a focus on frame structures.
"""

# Import and expose unit systems and other global constants
# from pyMAOS.globals import (
#     SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
#     FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT
# )

# Add references to the new modules
from .structure2d import R2Structure
from .structure2d_export import export_results_to_excel
from .structure2d_viz import plot_loadcombos_vtk
from .structure2d_utils import __str__

print("pyMAOS package initialized with unit systems and structural analysis tools.")

__all__ = [
    'SI_UNITS', 'IMPERIAL_UNITS', 'METRIC_KN_UNITS',
    'FORCE_UNIT', 'LENGTH_UNIT', 'MOMENT_UNIT', 'PRESSURE_UNIT', 'DISTRIBUTED_LOAD_UNIT',
    'R2Structure'  # The module itself
]
# Package version
__version__ = '0.1.0'