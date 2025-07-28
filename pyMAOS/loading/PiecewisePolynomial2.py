import numpy as np
from scipy.interpolate import PPoly
from pint import Quantity

from pyMAOS.units_mod import ureg


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
                    if isinstance(coeff, Quantity):
                        # Ensure consistent registry
                        if coeff._REGISTRY != ureg:
                            print(f"Converting coefficient {coeff} to unified form")
                            magnitude = coeff.magnitude
                            unit_str = str(coeff.units)
                            unified_coeffs.append(magnitude * ureg.parse_expression(unit_str))
                        else:
                            unified_coeffs.append(coeff)
                    else:
                        unified_coeffs.append(coeff)

                # Process interval
                unified_interval = []
                for point in interval:
                    if isinstance(point, Quantity):
                        if point._REGISTRY != ureg:
                            magnitude = point.magnitude
                            unit_str = str(point.units)
                            unified_interval.append(magnitude * ureg.parse_expression(unit_str))
                        else:
                            unified_interval.append(point)
                    else:
                        unified_interval.append(point)

                # Extract units
                segment_y_units = unified_coeffs[0].units if isinstance(unified_coeffs[0],
                                                                        Quantity) else ureg.dimensionless
                segment_x_units = unified_interval[0].units if isinstance(unified_interval[0],
                                                                          Quantity) else ureg.dimensionless

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

                # Build coefficient matrix for PPoly (scipy expects coeffs in descending order)
                c = np.zeros((len(all_coeffs[0]), len(breaks) - 1))

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
                            c[:len(magnitudes), i] = magnitudes[::-1]  # Reverse order for PPoly
                            break

                # Create the PPoly instance
                self.ppoly = PPoly(c, breaks)

                print(f"Created PiecewisePolynomial2 with y unit: {self.y_units} x unit: {self.x_units}")

        else:
            # Initialize with default units if no functions provided
            self.y_units = ureg.dimensionless
            self.x_units = ureg.dimensionless

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
        if self.x_units != ureg.dimensionless:
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
                    terms.append(f"{coef:.2f}")
                elif j == 1:
                    terms.append(f"{coef:.2f}x")
                else:
                    terms.append(f"{coef:.2f}x^{j}")

            poly_str = " + ".join(terms) if terms else "0"

            # Add segment with units
            if self.x_units != ureg.dimensionless:
                out.append(f"for {start:.2f} ≤ x ≤ {end:.2f} {self.x_units}: {poly_str} {self.y_units}")
            else:
                out.append(f"for {start:.2f} ≤ x ≤ {end:.2f}: {poly_str} {self.y_units}")

        return "\n".join(out)

    def __repr__(self):
        """Developer representation"""
        return f"PiecewisePolynomial2(functions={self.functions})"