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
# Configure numpy printing
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal')


# Add custom formatters for different numeric types
def format_with_dots(x) -> str: return '.'.center(12) if abs(x) < 1e-10 else f"{x:<12.4g}"


def format_double(x) -> str: return '.'.center(16) if abs(x) < 1e-10 else f"{x:<16.8g}"  # More precision for doubles


# np.set_printoptions(formatter={
#     np.float64: format_double,
#     np.float32: format_with_dots
# }) # type: ignore
np.set_printoptions(precision=2, threshold=999, linewidth=120, suppress=True,
                    formatter={np.float32: lambda x: '.'.center(10) if abs(x) < 1e-10 else f"{x:10.4g}"}) # type: ignore

print(np.get_printoptions())

import pprint

pp = pprint.PrettyPrinter(width=999, compact=False)

from pyMAOS.units_mod import UnitManager


# For SI internal units (default)
# unit_manager = UnitManager("SI")

# OR for Imperial internal units
unit_manager = UnitManager("imperial")

global FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY, LENGTH_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
# Add these definitions to the top of the file after initializing ureg
# Pre-computed dimensionality constants for efficient type checking
FORCE_DIMENSIONALITY = unit_manager.ureg.N.dimensionality
MOMENT_DIMENSIONALITY = (unit_manager.ureg.N * unit_manager.ureg.m).dimensionality
LENGTH_DIMENSIONALITY = unit_manager.ureg.m.dimensionality
PRESSURE_DIMENSIONALITY = unit_manager.ureg.Pa.dimensionality
DISTRIBUTED_LOAD_DIMENSIONALITY = (unit_manager.ureg.N / unit_manager.ureg.m).dimensionality



# For backward compatibility, define global variables that reference the UnitManager instance
global INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_TIME_UNIT, INTERNAL_ROTATION_UNIT, INTERNAL_DIMENSION_UNIT, INTERNAL_AREA_UNIT, INTERNAL_VOLUME_UNIT, INTERNAL_MOMENT_OF_INERTIA_UNIT, INTERNAL_DENSITY_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED, INTERNAL_DISTRIBUTED_LOAD_UNIT
INTERNAL_FORCE_UNIT = unit_manager.INTERNAL_FORCE_UNIT
INTERNAL_LENGTH_UNIT = unit_manager.INTERNAL_LENGTH_UNIT
INTERNAL_TIME_UNIT = unit_manager.INTERNAL_TIME_UNIT
INTERNAL_ROTATION_UNIT = unit_manager.INTERNAL_ROTATION_UNIT
INTERNAL_AREA_UNIT = unit_manager.INTERNAL_AREA_UNIT
INTERNAL_VOLUME_UNIT = unit_manager.INTERNAL_VOLUME_UNIT
INTERNAL_MOMENT_OF_INERTIA_UNIT = unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT
INTERNAL_DENSITY_UNIT = unit_manager.INTERNAL_DENSITY_UNIT
INTERNAL_MOMENT_UNIT = unit_manager.INTERNAL_MOMENT_UNIT
INTERNAL_PRESSURE_UNIT = unit_manager.INTERNAL_PRESSURE_UNIT
INTERNAL_PRESSURE_UNIT_EXPANDED = unit_manager.INTERNAL_PRESSURE_UNIT_EXPANDED
INTERNAL_DISTRIBUTED_LOAD_UNIT = unit_manager.INTERNAL_DISTRIBUTED_LOAD_UNIT

# Common imperial units as constants
global FOOT, INCH, POUND_FORCE, PSI, KSI
FOOT = unit_manager.ureg.foot
INCH = unit_manager.ureg.inch
POUND_FORCE = unit_manager.ureg.pound_force
PSI = unit_manager.ureg.psi
KSI = unit_manager.ureg.ksi

# Import and expose unit systems and other global constants
# from pyMAOS.units_mod import (SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
#     INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT,  INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT,
#     FORCE_DIMENSIONALITY, LENGTH_DIMENSIONALITY, MOMENT_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
# )

import pint
from pint import Quantity



# Add the method to the Quantity class
from pyMAOS.quantity_utils import increment_with_units
unit_manager.ureg.Quantity.increment_with_units = increment_with_units


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
    'R2Structure',  # The module itself
    'pint',  # Add pint to exports
    'pprint'  # Add pprint to exports
]
# Package version
__version__ = '0.1.0'