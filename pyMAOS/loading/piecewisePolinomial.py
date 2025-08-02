import math
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
import pint
from pint import Quantity
from typing import List, Union

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
from pyMAOS.units_mod import unit_manager
class PiecewisePolynomial:
    def __init__(self, functions=None):
        from pyMAOS.units_mod import unit_manager

        self.functions = []
        self.x_units = None
        self.y_units = None

        if functions:
            for coeffs_with_units, interval in functions:
                # Convert all coefficients to use our registry
                unified_coeffs = []
                for coeff in coeffs_with_units:
                    if isinstance(coeff, Quantity) and coeff._REGISTRY != unit_manager.ureg:
                            # print("Unit registry mismatch:")
                            # print(f"Coefficient units registry: {coeff._REGISTRY}")
                            # print(f"Global ureg registry: {unit_manager.ureg}")
                            magnitude = coeff.magnitude
                            unit_str = str(coeff.units)
                            unified_coeffs.append(magnitude * unit_manager.ureg.parse_expression(unit_str))
                    else:
                        unified_coeffs.append(coeff)

                # Do the same for interval points
                unified_interval = []
                for point in interval:
                    if isinstance(point, Quantity) and point._REGISTRY != unit_manager.ureg:
                        magnitude = point.magnitude
                        unit_str = str(point.units)
                        unified_interval.append(magnitude * unit_manager.ureg.parse_expression(unit_str))
                    else:
                        unified_interval.append(point)

                # Get units from unified quantities
                segment_y_units = unified_coeffs[0].units if isinstance(unified_coeffs[0], Quantity) else unit_manager.ureg.dimensionless
                segment_x_units = unified_interval[0].units if isinstance(unified_interval[0], Quantity) else unit_manager.ureg.dimensionless

                # If this is the first segment, set the class-level units
                if self.y_units is None:
                    # if segment_y_units.dimensionless:
                    #     raise Warning("Warning: Y units are dimensionless for the first segment.")
                    self.y_units = segment_y_units
                    self.x_units = segment_x_units
                else:
                    # Validate units consistency
                    if segment_y_units.dimensionality != self.y_units.dimensionality:
                        raise ValueError(f"Y units mismatch: {segment_y_units} vs {self.y_units}")
                    if segment_x_units.dimensionality != self.x_units.dimensionality:
                        raise ValueError(f"X units mismatch: {segment_x_units} vs {self.x_units}")

                try:
                    poly = UnitAwarePolynomial(unified_coeffs, y_units=self.y_units, x_units=self.x_units)
                    print(f"poly={poly}, interval={unified_interval}, y_units={self.y_units}, x_units={self.x_units}")
                except Exception as e:
                    print(f"Error creating UnitAwarePolynomial: {e}")
                    print(f"Coefficient types: {[type(c) for c in unified_coeffs]}")
                    print(f"Interval types: {[type(i) for i in unified_interval]}")
                    raise

                self.functions.append([poly, unified_interval, self.y_units, self.x_units])

            print(f"Created PiecewisePolynomial with y unit: {self.y_units} x unit: {self.x_units}")
        else:
            # Initialize with default units if no functions provided
            self.y_units = unit_manager.ureg.dimensionless
            self.x_units = unit_manager.ureg.dimensionless

    def evaluate(self, x):
        for poly, interval, y_units, x_units in self.functions:
            if interval[0] <= x <= interval[1]:
                # Use NumPy's polynomial evaluation
                return poly(x)
        return 0

    def evaluate_vectorized(self, x_array):
        """
        Evaluate the piecewise polynomial at multiple points simultaneously.
        Handles mixed unit types safely.

        Parameters
        ----------
        x_array : array_like
            Array of x values to evaluate

        Returns
        -------
        array_like
            Array of function values at each x
        """
        import numpy as np
        # if isinstance(x_array, pint.Quantity):
        #     # If x_array is a single Quantity, convert to numpy array
        #     x_array = np.array([x_array.magnitude]) * x_array.units
        # Convert to numpy array if not already
        if not isinstance(x_array, np.ndarray):
            x_array = np.array(x_array, dtype=object)

        # Initialize results array with proper units
        if not self.functions:
            return np.zeros_like(x_array)

        # Get units from first function's y_units if available
        if len(self.functions[0]) > 2:
            y_units = self.functions[0][2]
            results = np.zeros(x_array.shape, dtype=object)
            for i in range(len(results)):
                results[i] = 0 * y_units
        else:
            results = np.zeros_like(x_array)

        # For each point, find which function applies and evaluate
        for i, x in enumerate(x_array):
            for poly, interval, *unit_info in self.functions:
                # Safe comparison with unit handling
                lower_bound, upper_bound = interval

                # Convert values to compatible units for comparison
                x_compatible = x
                lower_compatible = lower_bound
                upper_compatible = upper_bound

                # Handle case where x has units but bounds don't
                if hasattr(x, 'units') and not hasattr(lower_bound, 'units'):
                    lower_compatible = lower_bound * x.units
                    upper_compatible = upper_bound * x.units

                # Handle case where bounds have units but x doesn't
                elif not hasattr(x, 'units') and hasattr(lower_bound, 'units'):
                    x_compatible = x * lower_bound.units

                # Now do the comparison with compatible units
                if lower_compatible <= x_compatible <= upper_compatible:
                    results[i] = poly(x)
                    break

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
                        new_coeffs[-(j+1)] += coeff

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
            result += coeff * x**i
        else:
            result += coeff * x**i
    return result