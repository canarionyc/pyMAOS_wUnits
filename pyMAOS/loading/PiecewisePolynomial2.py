import numpy as np
from scipy.interpolate import PPoly
from pint import Quantity

import pyMAOS
from pyMAOS import unit_manager


class PiecewisePolynomial2:
	"""
	Unit-aware piecewise polynomial class based on scipy.interpolate.PPoly.

	This class offers efficient evaluation and vectorized operations while
	maintaining unit consistency throughout calculations.
	"""

	def __init__(self, functions=None):
		"""
		Initialize a unit-aware piecewise polynomial.

		Parameters
		----------
		functions : list, optional
			List of [coeffs_with_units, interval] pairs, where:
			- coeffs_with_units is a list of coefficients with units
			- interval is [start, end] defining the domain of the polynomial
		"""
		self.functions = []
		self.x_units = None
		self.y_units = None
		self.ppoly = None  # Will hold the scipy PPoly instance

		if functions:
			# Process functions to get normalized data for PPoly creation
			breaks = []  # Break points
			all_coeffs = []  # Coefficients for each segment

			# First, process functions to extract units and validate consistency
			for coeffs_with_units, interval in functions:
				# Process coefficients
				unified_coeffs = []
				for coeff in coeffs_with_units:
					if isinstance(coeff, Quantity) and coeff._REGISTRY != unit_manager.ureg:
						print(f"Converting coefficient {coeff} to unified form")
						magnitude = coeff.magnitude
						unit_str = str(coeff.units)
						unified_coeffs.append(magnitude * unit_manager.ureg.parse_expression(unit_str))
					else:
						unified_coeffs.append(coeff)

				# Process interval
				unified_interval = []
				for point in interval:
					if isinstance(point, Quantity) and point._REGISTRY != unit_manager.ureg:
						magnitude = point.magnitude
						unit_str = str(point.units)
						unified_interval.append(magnitude * unit_manager.ureg.parse_expression(unit_str))
					else:
						unified_interval.append(point)

				# Extract units
				segment_y_units = unified_coeffs[0].units if isinstance(unified_coeffs[0],
				                                                        Quantity) else unit_manager.ureg.dimensionless
				segment_x_units = unified_interval[0].units if isinstance(unified_interval[0],
				                                                          Quantity) else unit_manager.ureg.dimensionless

				# Set or validate units consistency
				if self.y_units is None:
					self.y_units = segment_y_units
					self.x_units = segment_x_units
				else:
					# Validate units consistency
					if segment_y_units.dimensionality != self.y_units.dimensionality:
						raise ValueError(f"Y units mismatch: {segment_y_units} vs {self.y_units}")
					if segment_x_units.dimensionality != self.x_units.dimensionality:
						raise ValueError(f"X units mismatch: {segment_x_units} vs {self.x_units}")

				# Store processed function
				start, end = unified_interval
				breaks.extend([start.magnitude if isinstance(start, Quantity) else start,
				               end.magnitude if isinstance(end, Quantity) else end])

				# Extract magnitudes for PPoly creation
				magnitudes = [c.magnitude if isinstance(c, Quantity) else c for c in unified_coeffs]
				all_coeffs.append(magnitudes)

				# Store original data for reference
				self.functions.append([unified_coeffs, unified_interval, self.y_units, self.x_units])

			# Create PPoly instance from processed data
			if all_coeffs:
				# Sort breaks and remove duplicates
				breaks = sorted(set(breaks))

				# Find the maximum degree across all polynomials
				max_degree = max(len(coeffs) for coeffs in all_coeffs)

				# Build coefficient matrix for PPoly (scipy expects coeffs in descending order)
				c = np.zeros((max_degree, len(breaks) - 1))

				# For each segment between breakpoints
				for i in range(len(breaks) - 1):
					start, end = breaks[i], breaks[i + 1]

					# Find polynomial that applies to this segment
					for coeffs, interval, _, _ in self.functions:
						interval_start = interval[0].magnitude if isinstance(interval[0], Quantity) else interval[0]
						interval_end = interval[1].magnitude if isinstance(interval[1], Quantity) else interval[1]

						if interval_start <= start and interval_end >= end:
							# Extract magnitudes and add to coefficient matrix
							magnitudes = [coef.magnitude if isinstance(coef, Quantity) else coef for coef in coeffs]
							# Reverse and pad with zeros if needed
							padded_magnitudes = magnitudes[::-1] + [0] * (max_degree - len(magnitudes))
							c[:, i] = padded_magnitudes  # Assign to column
							break

				print(f"Debug - c matrix shape: {c.shape}, breaks shape: {len(breaks)}")
				# Create the PPoly instance
				self.ppoly = PPoly(c, breaks)

				print(f"Created PiecewisePolynomial2 with y unit: {self.y_units} x unit: {self.x_units}")

		else:
			# Initialize with default units if no functions provided
			self.y_units = unit_manager.ureg.dimensionless
			self.x_units = unit_manager.ureg.dimensionless

	def evaluate(self, x):
		"""
		Evaluate the piecewise polynomial at point x.

		Parameters
		----------
		x : scalar or Quantity
			Point at which to evaluate the polynomial

		Returns
		-------
		scalar or Quantity
			Function value at x with appropriate units
		"""
		if self.ppoly is None:
			return 0

		# Handle unit conversion if x has units
		if isinstance(x, Quantity):
			# Check unit compatibility
			if x.dimensionality != self.x_units.dimensionality:
				raise ValueError(f"Input x units {x.units} not compatible with polynomial x units {self.x_units}")
			# Extract magnitude in the polynomial's x units
			x_magnitude = x.to(self.x_units).magnitude
		else:
			# No units provided - use as is
			x_magnitude = x

		# Use PPoly for evaluation
		result_magnitude = self.ppoly(x_magnitude)

		# Apply units to result
		return result_magnitude * self.y_units

	def evaluate_vectorized(self, x_array):
		"""
		Evaluate the piecewise polynomial at multiple points simultaneously.

		Parameters
		----------
		x_array : array_like or Quantity array
			Array of x values to evaluate

		Returns
		-------
		array_like with units
			Array of function values at each x with appropriate units
		"""
		if self.ppoly is None:
			return np.zeros_like(x_array)

		# Convert to numpy array if not already
		is_quantity = isinstance(x_array, Quantity)

		if is_quantity:
			# Check unit compatibility
			if x_array.dimensionality != self.x_units.dimensionality:
				raise ValueError(f"Input x units {x_array.units} not compatible with polynomial x units {self.x_units}")
			# Extract magnitudes
			x_magnitudes = x_array.to(self.x_units).magnitude
		else:
			# Convert to numpy array if not already
			x_magnitudes = np.asarray(x_array)

		# Use PPoly for vectorized evaluation
		result_magnitudes = self.ppoly(x_magnitudes)

		# Apply units to results
		return result_magnitudes * self.y_units

	def roots(self):
		"""
		Find roots of the piecewise polynomial within its domain.

		Returns
		-------
		list
			Sorted list of x values where the polynomial equals zero
		"""
		if self.ppoly is None:
			return []

		all_roots = []
		for i in range(len(self.ppoly.x) - 1):
			# Get polynomial coefficients for this segment
			c = self.ppoly.c[:, i]

			# Find roots in this segment
			segment_roots = np.roots(c[::-1])  # Reverse order for np.roots

			# Filter roots within segment domain
			valid_roots = []
			for root in segment_roots:
				if root.imag == 0:  # Only consider real roots
					x = root.real
					if self.ppoly.x[i] <= x <= self.ppoly.x[i + 1]:
						valid_roots.append(x)

			all_roots.extend(valid_roots)

		# Add units to roots if needed
		if self.x_units != unit_manager.ureg.dimensionless:
			all_roots = [r * self.x_units for r in all_roots]

		return sorted(all_roots)

	def __str__(self):
		"""String representation showing each polynomial segment"""
		if not self.functions or self.ppoly is None:
			return "No piecewise polynomial functions defined."

		out = []
		for i in range(len(self.ppoly.x) - 1):
			start, end = self.ppoly.x[i], self.ppoly.x[i + 1]
			coeffs = self.ppoly.c[:, i][::-1]  # Reverse to get ascending power

			# Format polynomial expression
			terms = []
			for j, coef in enumerate(coeffs):
				if coef == 0:
					continue
				if j == 0:
					terms.append(f"{coef:+.2g}")
				elif j == 1:
					terms.append(f"{coef:+.2g}x")
				else:
					terms.append(f"{coef:+.2g}x^{j}")

			poly_str = " ".join(terms) if terms else "0"

			# Add segment with units
			if self.x_units != unit_manager.ureg.dimensionless:
				out.append(f"for {start:.2f} ≤ x ≤ {end:.2f} {self.x_units}: {poly_str} {self.y_units}")
			else:
				out.append(f"for {start:.2f} ≤ x ≤ {end:.2f}: {poly_str} {self.y_units}")

		return "\n".join(out)

	def __repr__(self):
		"""Developer representation"""
		return f"PiecewisePolynomial2(functions={self.functions})"

	def plot(self, ax=None, num_points=100, color='blue', title=None,
	         convert_x_to=None, convert_y_to=None, linewidth=2, show=True):
		"""
		Plot the piecewise polynomial using the efficient PPoly representation.

		Parameters
		----------
		ax : matplotlib.axes.Axes, optional
			Axes to plot on. If None, creates a new figure and axes.
		num_points : int, optional
			Number of points to use for plotting
		color : str, optional
			Line color
		title : str, optional
			Plot title
		convert_x_to : pint.Unit, optional
			Unit to convert x-values to for plotting
		convert_y_to : pint.Unit, optional
			Unit to convert y-values to for plotting
		linewidth : float, optional
			Width of the plot line
		show : bool, optional
			Whether to show the plot immediately

		Returns
		-------
		matplotlib.axes.Axes
			The axes containing the plot
		"""
		import matplotlib.pyplot as plt
		import numpy as np

		# Create axes if not provided
		if ax is None:
			fig, ax = plt.subplots(figsize=(10, 6))

		if self.ppoly is None:
			ax.text(0.5, 0.5, "No function defined", ha='center', va='center')
			return ax

		# Generate x values across the entire domain
		x_min, x_max = self.ppoly.x[0], self.ppoly.x[-1]
		x_values = np.linspace(x_min, x_max, num_points)

		# Evaluate function at these points
		y_values = self.ppoly(x_values)

		# Apply units to results
		if self.x_units != unit_manager.ureg.dimensionless:
			x_values = x_values * self.x_units
		if self.y_units != unit_manager.ureg.dimensionless:
			y_values = y_values * self.y_units

		# Handle unit conversion for plotting
		if convert_x_to and hasattr(x_values, 'to'):
			x_values = x_values.to(convert_x_to).magnitude
		elif hasattr(x_values, 'magnitude'):
			x_values = x_values.magnitude

		if convert_y_to and hasattr(y_values, 'to'):
			y_values = y_values.to(convert_y_to).magnitude
		elif hasattr(y_values, 'magnitude'):
			y_values = y_values.magnitude

		# Plot the function
		ax.plot(x_values, y_values, color=color, linewidth=linewidth)

		# Add domain markers at breakpoints
		for breakpoint in self.ppoly.x:
			if hasattr(breakpoint, 'magnitude'):
				breakpoint = breakpoint.magnitude
			if convert_x_to:
				# If the breakpoint has units, we need to convert it
				bp = breakpoint * self.x_units if self.x_units != unit_manager.ureg.dimensionless else breakpoint
				if hasattr(bp, 'to'):
					bp = bp.to(convert_x_to).magnitude
				ax.axvline(x=bp, color='gray', linestyle=':', alpha=0.7, linewidth=1)
			else:
				ax.axvline(x=breakpoint, color='gray', linestyle=':', alpha=0.7, linewidth=1)

		# Add labels
		ax.set_xlabel(f"x [{convert_x_to if convert_x_to else self.x_units}]")
		ax.set_ylabel(f"y [{convert_y_to if convert_y_to else self.y_units}]")

		# Add title if provided
		if title:
			ax.set_title(title)

		# Add grid
		ax.grid(True, linestyle='--', alpha=0.7)

		if show:
			plt.tight_layout()
			plt.show()

		return ax
