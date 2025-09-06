"""
PyMAOS Structural Loading Module

This module provides classes for representing different types of loads applied
to structural elements, including point loads, distributed loads, and moments.
"""

import pint
import pprint
from pprint import pprint
from pyMAOS.loading.piecewisePolinomial import polynomial_evaluation, PiecewisePolynomial
from pyMAOS.loading.point_loads import R2_Point_Load, R2_Point_Moment
from pyMAOS.loading.distributed_loads import LinearLoadXY
from pyMAOS.loading.axial_loads import R2_Axial_Load, R2_Axial_Linear_Load

# Legacy module variables for backward compatibility
__all__ = [
	'polynomial_evaluation',
	'PiecewisePolynomial',
	'R2_Point_Load',
	'R2_Point_Moment',
	'LinearLoadXY',
	'R2_Axial_Load',
	'R2_Axial_Linear_Load'
]
