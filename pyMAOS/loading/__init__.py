"""
PyMAOS Structural Loading Module

This module provides classes for representing different types of loads applied
to structural elements, including point loads, distributed loads, and moments.
"""

from pyMAOS.loading.polynomial import polynomial_evaluation, Piecewise_Polynomial
from pyMAOS.loading.point_loads import R2_Point_Load, R2_Point_Moment
from pyMAOS.loading.distributed_loads import R2_Linear_Load
from pyMAOS.loading.axial_loads import R2_Axial_Load, R2_Axial_Linear_Load

# Legacy module variables for backward compatibility
__all__ = [
    'polynomial_evaluation',
    'Piecewise_Polynomial',
    'R2_Point_Load',
    'R2_Point_Moment',
    'R2_Linear_Load',
    'R2_Axial_Load',
    'R2_Axial_Linear_Load',
]
