"""
pyMAOS - Python Mechanics and Analysis of Structures

This package provides tools for structural engineering analysis,
with a focus on frame structures.
"""
import os
import sys
# Print module search paths and loading information
print("\n=== Python Module Search Paths ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nModule search paths (sys.path):")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")
print("=" * 40 + "\n")

import numpy as np
print(np.get_printoptions())

import pprint

pp = pprint.PrettyPrinter(width=999, compact=False)

# Import and expose unit systems and other global constants
from pyMAOS.units_mod import (SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
    INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT,
    FORCE_DIMENSIONALITY, LENGTH_DIMENSIONALITY, MOMENT_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
)




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
    'FORCE_DIMENSIONALITY', 'LENGTH_DIMENSIONALITY', 'MOMENT_DIMENSIONALITY', 'PRESSURE_DIMENSIONALITY', 'DISTRIBUTED_LOAD_DIMENSIONALITY',
    'R2Structure'  # The module itself
    'pint',  # Add pint to exports
    'pprint'  # Add pprint to exports
]
# Package version
__version__ = '0.1.0'