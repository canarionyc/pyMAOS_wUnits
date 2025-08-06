"""
pyMAOS - Python Mechanics and Analysis of Structures

This package provides tools for structural engineering analysis,
with a focus on frame structures.
"""
import os

# Import essential modules first
from . import logger

# Create package-level logger objects for easy import
log = logger.default_logger

# Provide convenience functions at package level
def debug(msg, *args, **kwargs):
    log.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    log.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    log.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    log.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    log.critical(msg, *args, **kwargs)

def log_exception(message="An exception occurred", exc_info=None):
    logger.log_exception(log, exc_info, message)

# Import other modules after logger is set up
# ... other imports ...

if False:
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

if False:
    print(np.get_printoptions())

import pprint

pp = pprint.PrettyPrinter(width=999, compact=False)


from pyMAOS.pymaos_units import UnitManager


# For SI internal units (default)
# unit_manager = UnitManager("SI")

cache_folder= os.path.expanduser("~/.cache/pyMAOS")
unit_manager = UnitManager("imperial",cache_folder=cache_folder)

# Common imperial units as constants
global FOOT, INCH, POUND_FORCE, PSI, KSI
FOOT = unit_manager.ureg.foot
INCH = unit_manager.ureg.inch
POUND_FORCE = unit_manager.ureg.pound_force
PSI = unit_manager.ureg.psi
KSI = unit_manager.ureg.ksi

global FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY, LENGTH_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
# Add these definitions to the top of the file after initializing ureg
# Pre-computed dimensionality constants for efficient type checking
FORCE_DIMENSIONALITY = unit_manager.ureg.N.dimensionality
MOMENT_DIMENSIONALITY = (unit_manager.ureg.N * unit_manager.ureg.m).dimensionality
LENGTH_DIMENSIONALITY = unit_manager.ureg.m.dimensionality
PRESSURE_DIMENSIONALITY = unit_manager.ureg.Pa.dimensionality
DISTRIBUTED_LOAD_DIMENSIONALITY = (unit_manager.ureg.N / unit_manager.ureg.m).dimensionality

# For backward compatibility, define global variables that reference the UnitManager instance
global INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_TIME_UNIT, INTERNAL_ROTATION_UNIT, INTERNAL_DIMENSION_UNIT, INTERNAL_AREA_UNIT, INTERNAL_VOLUME_UNIT, INTERNAL_MOMENT_OF_INERTIA_UNIT, INTERNAL_DENSITY_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED, INTERNAL_DISTRIBUTED_LOAD_UNIT, INTERNAL_MASS_UNIT
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
INTERNAL_MASS_UNIT=unit_manager.INTERNAL_MASS_UNIT

# Add this function to pyMAOS/__init__.py after the global variables are defined
def _update_global_units(system_dict=None):
    """Update all global unit variables from the unit manager"""
    global INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_TIME_UNIT
    global INTERNAL_ROTATION_UNIT, INTERNAL_AREA_UNIT, INTERNAL_VOLUME_UNIT
    global INTERNAL_MOMENT_OF_INERTIA_UNIT, INTERNAL_DENSITY_UNIT
    global INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT
    global INTERNAL_PRESSURE_UNIT_EXPANDED, INTERNAL_DISTRIBUTED_LOAD_UNIT, INTERNAL_MASS_UNIT

    # Force update of derived units in unit manager
    unit_manager._update_derived_units()

    INTERNAL_MASS_UNIT=unit_manager.INTERNAL_MASS_UNIT
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

    print(f"DEBUG: Updated global unit variables to match system: {unit_manager.system_name}")

# Register the update function with unit manager
unit_manager.register_for_unit_updates(_update_global_units)



# Import and expose unit systems and other global constants
# from pyMAOS.units_mod import (SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
#     INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT,  INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT,
#     FORCE_DIMENSIONALITY, LENGTH_DIMENSIONALITY, MOMENT_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY
# )

import pint

# Add the method to the Quantity class
from pyMAOS.quantity_utils import increment_with_units
unit_manager.ureg.Quantity.increment_with_units = increment_with_units

# Add references to the new modules
from .structure2d import R2Structure

print("pyMAOS package initialized with unit systems and structural analysis tools.")

__all__ = [
    'SI_UNITS', 'IMPERIAL_DISPLAY_UNITS', 'METRIC_KN_UNITS',
    'INTERNAL_FORCE_UNIT', 'INTERNAL_LENGTH_UNIT', 'INTERNAL_MOMENT_UNIT', 'INTERNAL_PRESSURE_UNIT', 'INTERNAL_DISTRIBUTED_LOAD_UNIT',
    'FORCE_DIMENSIONALITY', 'LENGTH_DIMENSIONALITY', 'MOMENT_DIMENSIONALITY', 'PRESSURE_DIMENSIONALITY', 'DISTRIBUTED_LOAD_DIMENSIONALITY',
    'R2Structure',  # The module itself
    'pint',  # Add pint to exports
    'pprint'  # Add pprint to exports
    'debug','info','warning','error','critical','log_exception'
]
# Package version
__version__ = '0.1.0'