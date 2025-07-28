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
from pyMAOS.units_mod import ureg
class PiecewisePolynomial:
    def __init__(self, functions=None):
        from pyMAOS.units_mod import ureg

        self.functions = []
        self.x_units = None
        self.y_units = None

        if functions:
            for coeffs_with_units, interval in functions:
                # Convert all coefficients to use our registry
                unified_coeffs = []
                for coeff in coeffs_with_units:
                    if isinstance(coeff, Quantity):
                        print(f"Converting coefficient {coeff} to unified form")
                        if(coeff._REGISTRY != ureg):
                            print("Unit registry mismatch:")
                            print(f"Coefficient units registry: {coeff.units._REGISTRY}")
                            print(f"Global ureg registry: {ureg}")
                            magnitude = coeff.magnitude
                            unit_str = str(coeff.units)
                            unified_coeffs.append(magnitude * ureg.parse_expression(unit_str))
                        else:
                            unified_coeffs.append(coeff)
                    else:
                        unified_coeffs.append(coeff)

                # Do the same for interval points
                unified_interval = []
                for point in interval:
                    if isinstance(point, Quantity):
                        magnitude = point.magnitude
                        unit_str = str(point.units)
                        unified_interval.append(magnitude * ureg.parse_expression(unit_str))
                    else:
                        unified_interval.append(point)

                # Get units from unified quantities
                segment_y_units = unified_coeffs[0].units if isinstance(unified_coeffs[0], Quantity) else ureg.dimensionless
                segment_x_units = unified_interval[0].units if isinstance(unified_interval[0], Quantity) else ureg.dimensionless

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
            self.y_units = ureg.dimensionless
            self.x_units = ureg.dimensionless

    def evaluate(self, x):
        for poly, interval, y_units, x_units in self.functions:
            if interval[0] <= x <= interval[1]:
                # Use NumPy's polynomial evaluation
                return poly(x)
        return 0

    def evaluate_vectorized(self, x_array):
        """
        Evaluate the piecewise polynomial at multiple points simultaneously.

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
                if interval[0] <= x <= interval[1]:
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

    # def evaluate(self, x):
    #     """
    #     Given a piecewise function and an x evaluate the results
    #     """
    #     # in the context of the beam model a tolerance of 1E-6 will
    #     # yield acceptable results as we are evaluating normal polynomials
    #     tol = 0.000001
    #
    #     # initialize res to avoid an ref before assignment error in
    #     # the case where the below reaches pass for all conditions.
    #     piece_function = self.functions
    #     res = 0
    #
    #     if piece_function == []:
    #         res = 0
    #     else:
    #         for line in piece_function:
    #             if (line[1][0] - tol) < x <= (line[1][1] + tol):
    #                 res = polynomial_evaluation(line[0], x)
    #             else:
    #                 # x is not in the current functions range
    #                 pass
    #     return res

    # def roots(self):
    #     """
    #     Given a piecewise function return a list
    #     of the location of zeros or sign change
    #     """
    #     piece_function = self.functions
    #     zero_loc = []
    #     i = 0
    #     for line in piece_function:
    #         if len(line[0]) == 1 and i == 0:
    #             pass  # If function is a value then there is no chance for a sign change
    #         else:
    #             a = polynomial_evaluation(
    #                 line[0], line[1][0] + 0.0001
    #             )  # value at start of bounds
    #             b = polynomial_evaluation(
    #                 line[0], line[1][1] - 0.0001
    #             )  # value at end of bounds
    #
    #             if a == 0:
    #                 zero_loc.append(line[1][0])
    #             elif b == 0:
    #                 zero_loc.append(line[1][1])
    #             else:
    #                 # if signs are the the same a/b will result in a positive value
    #                 coeff = line[0][::-1]
    #                 c = np.roots(coeff)
    #                 # Some real solutions may contain a very small imaginary part
    #                 # account for this with a tolerance on the imaginary
    #                 # part of 1e-5
    #                 c = c.real[abs(c.imag) < 1e-5]
    #                 for root in c:
    #                     # We only want roots that are with the piece range
    #                     if line[1][0] < root <= line[1][1]:
    #                         zero_loc.append(root)
    #                     else:
    #                         pass
    #             if i == 0:
    #                 pass
    #             else:
    #                 # value at end of previous bounds
    #                 d = polynomial_evaluation(
    #                     piece_function[i - 1][0], line[1][0] - 0.0001
    #                 )
    #
    #                 if d == 0:
    #                     pass
    #                 elif a / d < 0:
    #                     zero_loc.append(line[1][0])
    #                 else:
    #                     pass
    #         i += 1
    #     zero_loc = sorted(set(zero_loc))
    #     return zero_loc

    # def combine(self, other, LF, LFother):
    #     """
    #     Join two piecewise functions to create one piecewise function ecompassing
    #     the ranges and polynomials associated with each
    #     """
    #     Fa = self.functions
    #     Fb = other.functions
    #     LFa = LF
    #     LFb = LFother
    #
    #     functions = [Fa, Fb]
    #     LF = [LFa, LFb]
    #
    #     # Gather the ranges for each piece of the the two input functions
    #     ab = []
    #     for func in Fa:
    #         ab.append(func[1][0])
    #         ab.append(func[1][1])
    #     for func in Fb:
    #         ab.append(func[1][0])
    #         ab.append(func[1][1])
    #     ab = list(set(ab))
    #     ab.sort()
    #
    #     f_out = []
    #
    #     for i, j in enumerate(ab):
    #         if i == 0:
    #             piece_range = [0, j]
    #         else:
    #             piece_range = [ab[i - 1], j]
    #         if piece_range == [0, 0]:
    #             pass
    #         else:
    #             f = []
    #
    #             for i, func in enumerate(functions):
    #                 for piece in func:
    #                     if (
    #                         piece[1][0] < piece_range[1]
    #                         and piece[1][1] >= piece_range[1]
    #                     ):
    #                         # difference in number of coefficients
    #                         eq_len_delta = len(piece[0]) - len(f)
    #
    #                         if eq_len_delta > 0:
    #                             f.extend([0] * eq_len_delta)
    #                         elif eq_len_delta < 0:
    #                             piece[0].extend([0] * abs(eq_len_delta))
    #                         else:
    #                             pass
    #                         f = [j * LF[i] + k for j, k in zip(piece[0], f)]
    #                     else:
    #                         pass
    #             f_out.append([f, piece_range])
    #     return PiecewisePolynomial(f_out)
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

def evaluate_with_units(poly, x):
    """Evaluate a NumPy polynomial with unit-aware coefficients"""
    result = 0
    for i, coeff in enumerate(poly.coef):
        if isinstance(coeff, Quantity):
            result += coeff * x**i
        else:
            result += coeff * x**i
    return result