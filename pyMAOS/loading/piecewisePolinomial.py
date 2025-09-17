import math
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
import pint
from pint import Quantity
from typing import List, Union
import pyMAOS
from pyMAOS import unit_manager
from pyMAOS.loading.UnitAwarePolynomial import UnitAwarePolynomial


# from pyMAOS.units_mod import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS
# from pyMAOS.unit_aware import UnitAwareMixin
def polynomial_evaluation(c_list, x):
	"""
	evaluate a polynomial defined by a list of coeff. in ascending order
	C0 + C1x + C2x^2 + ... + Cnx^n = [C0,C1,C2,...,Cn]
	"""
	i = 0
	res = 0
	if all(c == 0 for c in c_list):
		pass
	else:
		for c in c_list:
			res = res + c * math.pow(x, i)
			i += 1
	return res


from numpy.polynomial import Polynomial
from pyMAOS.loading.UnitAwarePolynomial import UnitAwarePolynomial


#%% Improved PiecewisePolynomial class structure
class PiecewisePolynomial:
	def __init__(self, functions=None, x_units=None, y_units=None):
		"""
		Initialize piecewise polynomial with explicit unit specification
		"""
		# Set default units
		self.x_units = x_units or unit_manager.ureg.dimensionless
		self.y_units = y_units or unit_manager.ureg.dimensionless

		# Store segments as sorted list for efficient searching
		self.segments = []  # List of (interval_start, interval_end, polynomial)

		if functions:
			self._add_functions(functions)

	def _add_functions(self, functions):
		"""Add multiple function segments with validation"""
		for coeffs_with_units, interval in functions:
			self.add_segment(coeffs_with_units, interval)

		# Sort segments by start point
		self.segments.sort(key=lambda x: x[0])

		# Validate no overlaps
		self._validate_segments()

	def _validate_segments(self):
		"""Ensure segments don't overlap"""
		for i in range(len(self.segments) - 1):
			if self.segments[i][1] > self.segments[i+1][0]:
				raise ValueError(f"Overlapping segments detected between {self.segments[i]} and {self.segments[i+1]}")

	def evaluate_vectorized(self, x_array):
		"""
		Vectorized evaluation using binary search for efficiency
		"""
		# Ensure unit consistency
		if isinstance(x_array, Quantity):
			x_array = x_array.to(self.x_units)
		elif self.x_units != unit_manager.ureg.dimensionless:
			x_array = x_array * self.x_units

		# Extract interval boundaries for efficient searching
		boundaries = [seg[0].magnitude if isinstance(seg[0], Quantity) else seg[0]
					 for seg in self.segments]

		# Initialize results
		results = np.zeros(len(x_array)) * self.y_units

		# Use searchsorted to find which segment each x belongs to
		x_magnitudes = x_array.magnitude if isinstance(x_array, Quantity) else x_array
		indices = np.searchsorted(boundaries, x_magnitudes, side='right') - 1

		# Evaluate each segment
		for i, (start, end, poly) in enumerate(self.segments):
			mask = (indices == i) & (x_magnitudes <= end.magnitude if isinstance(end, Quantity) else end)
			if np.any(mask):
				results[mask] = poly(x_array[mask])

		return results
	def roots(self):
		all_roots = []
		for poly, interval in self.functions:
			# Use NumPy's root finding with filtering for the interval
			r = poly.roots()
			valid_roots = [root for root in r if interval[0] <= root <= interval[1]]
			all_roots.extend(valid_roots)
		return sorted(all_roots)

	def __str__(self):
		out = ""
		if not self.functions:
			return "No piecewise polynomial functions defined."
		# from pprint import pprint;
		# pprint(self.functions)
		for line in self.functions:
			# if not has_term:
			#     func = "0"

			func = f"for {line[1][0]:.2f} ≤ x ≤ {line[1][1]:.2f}: "
			if isinstance(line[0], UnitAwarePolynomial):
				func += line[0].__str__()
			elif isinstance(line[0], Polynomial):
				func += " + ".join(
					f"{coef:.2f}x^{i}" for i, coef in enumerate(line[0].coef)
				)

			# If the function is just zero, display 0

			out += func + "\n"

		return out.rstrip("\n")

	def __repr__(self):
		return f"PiecewisePolynomial(functions={self.functions})"

	def combine(self, other, LF=1, LFother=1):
		"""Combine two piecewise polynomials with load factors

		Parameters
		----------
		other : PiecewisePolynomial
			Another piecewise polynomial to combine with this one
		LF : float
			Load factor to apply to this polynomial (default: 1)
		LFother : float
			Load factor to apply to the other polynomial (default: 1)

		Returns
		-------
		PiecewisePolynomial
			A new piecewise polynomial that combines both inputs with their load factors

		Raises
		------
		TypeError
			If other is not a Piecewise_Polynomial
		"""
		if not isinstance(other, PiecewisePolynomial):
			raise TypeError("Can only combine with another PiecewisePolynomial")

		result = PiecewisePolynomial()

		# Add scaled segments from this polynomial
		for poly, interval in self.functions:
			start, end = interval
			scaled_coeffs = [coeff * LF for coeff in poly.coef]
			result.add_segment(start, end, scaled_coeffs)

		# Add scaled segments from the other polynomial
		for poly, interval in other.functions:
			start, end = interval
			scaled_coeffs = [coeff * LFother for coeff in poly.coef]
			result.add_segment(start, end, scaled_coeffs)

		# Merge overlapping intervals
		result.merge_segments()

		return result

	def add_segment(self, start, end, coeffs):
		"""Add a polynomial segment to the piecewise function

		Parameters
		----------
		start : float
			Start of the interval
		end : float
			End of the interval
		coeffs : list
			List of polynomial coefficients [a_n, ..., a_1, a_0]
			where p(x) = a_n*x^n + ... + a_1*x + a_0
		"""
		if start >= end:
			raise ValueError("Start must be less than end")

		poly = Polynomial(coeffs)
		self.functions.append([poly, [start, end]])

	# You may need to adjust the merge_segments method depending on how your Polynomial class stores coefficients (highest order first or last).
	def merge_segments(self):
		"""Merge overlapping segments by breaking them at intersection points"""
		if not self.functions:
			return

		# Find all unique boundaries
		boundaries = set()
		for _, interval in self.functions:
			boundaries.add(interval[0])
			boundaries.add(interval[1])

		boundaries = sorted(boundaries)

		# Create new segments based on these boundaries
		new_functions = []
		for i in range(len(boundaries) - 1):
			start = boundaries[i]
			end = boundaries[i + 1]

			# If start equals end, skip this interval
			if start == end:
				continue

			# Find all functions that apply in this interval
			applicable_polys = []
			for poly, interval in self.functions:
				if interval[0] <= start and interval[1] >= end:
					applicable_polys.append(poly)

			# If there are applicable functions, combine them
			if applicable_polys:
				# Create a new polynomial that is the sum of all applicable polynomials
				new_coeffs = [0] * max(len(poly.coeffs) for poly in applicable_polys)
				for poly in applicable_polys:
					for j, coeff in enumerate(reversed(poly.coeffs)):
						new_coeffs[-(j + 1)] += coeff

				new_poly = Polynomial(list(reversed(new_coeffs)))
				new_functions.append([new_poly, [start, end]])

		self.functions = new_functions

	def plot(self, ax=None, num_points=1000, x_label=None, y_label=None,
	         color='blue', linestyle='-', linewidth=2, alpha=1.0,
	         convert_x_to=None, convert_y_to=None, title=None, legend=True, show=True):
		"""
		Plot the piecewise polynomial function using matplotlib.

		Parameters
		----------
		ax : matplotlib.axes.Axes, optional
			Axes to plot on. If None, creates a new figure and axes.
		num_points : int, optional
			Number of points to use for the plot (default: 1000)
		x_label : str, optional
			Label for x-axis. If None, uses x units.
		y_label : str, optional
			Label for y-axis. If None, uses y units.
		color : str, optional
			Line color (default: 'blue')
		linestyle : str, optional
			Line style (default: '-')
		linewidth : float, optional
			Line width (default: 2)
		alpha : float, optional
			Line transparency (default: 1.0)
		convert_x_to : pint.Unit, optional
			Convert x values to this unit for plotting
		convert_y_to : pint.Unit, optional
			Convert y values to this unit for plotting
		title : str, optional
			Plot title

		Returns
		-------
		matplotlib.axes.Axes
			The axes containing the plot

		Notes
		-----
		If the polynomial has units, they will be displayed in the axis labels
		unless custom labels are provided.
		"""
		import matplotlib.pyplot as plt
		import numpy as np

		if not self.functions:
			print("No polynomial segments to plot")
			return None

		# Create new figure if needed
		if ax is None:
			fig, ax = plt.subplots(figsize=(10, 6))

		# Find the full domain range
		x_min = min(interval[0] for _, interval, *_ in self.functions)
		x_max = max(interval[1] for _, interval, *_ in self.functions)

		# Generate x points
		if isinstance(x_min, Quantity):
			# Create unit-aware array
			x_values = np.linspace(x_min.magnitude, x_max.magnitude, num_points)
			x_values = x_values * x_min.units

			if convert_x_to is not None:
				x_plot = x_values.to(convert_x_to).magnitude
				x_unit_str = f" ({convert_x_to})"
			else:
				x_plot = x_values.magnitude
				x_unit_str = f" ({x_min.units})"
		else:
			x_values = np.linspace(x_min, x_max, num_points)
			x_plot = x_values
			x_unit_str = ""

		# Evaluate y values
		y_values = self.evaluate_vectorized(x_values)

		# Handle y units for plotting
		if isinstance(y_values[0], Quantity):
			if convert_y_to is not None:
				y_plot = np.array([y.to(convert_y_to).magnitude for y in y_values])
				y_unit_str = f" ({convert_y_to})"
			else:
				y_plot = np.array([y.magnitude for y in y_values])
				y_unit_str = f" ({y_values[0].units})"
		else:
			y_plot = y_values
			y_unit_str = ""

		# Plot the function
		ax.plot(x_plot, y_plot, color=color, linestyle=linestyle,
		        linewidth=linewidth, alpha=alpha)

		# Add labels
		if x_label is None:
			ax.set_xlabel(f"x{x_unit_str}")
		else:
			ax.set_xlabel(x_label)

		if y_label is None:
			ax.set_ylabel(f"f(x){y_unit_str}")
		else:
			ax.set_ylabel(y_label)

		if title is not None:
			ax.set_title(title)

		# Add grid for better readability
		ax.grid(True, linestyle='--', alpha=0.7)

		# Mark interval boundaries
		interval_points = sorted(set([interval[0] for _, interval, *_ in self.functions] +
		                             [interval[1] for _, interval, *_ in self.functions]))

		for point in interval_points:
			if isinstance(point, Quantity):
				if convert_x_to is not None:
					point_plot = point.to(convert_x_to).magnitude
				else:
					point_plot = point.magnitude
			else:
				point_plot = point

			# Plot small vertical lines at interval boundaries
			ax.axvline(x=point_plot, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

		# Ensure tight layout
		plt.tight_layout()

		return ax


def evaluate_with_units(poly, x):
	"""Evaluate a NumPy polynomial with unit-aware coefficients"""
	result = 0
	for i, coeff in enumerate(poly.coef):
		if isinstance(coeff, Quantity):
			result += coeff * x ** i
		else:
			result += coeff * x ** i
	return result
