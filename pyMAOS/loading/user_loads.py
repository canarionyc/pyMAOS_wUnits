import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam
import pint
from typing import Any, Callable, Union, Optional
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial

class R2_SymPy_Load(R2_Load_Base):
    """Class for user-defined loads using SymPy symbolic expressions"""

    def __init__(self, load_func: Union[Callable, sp.Expr],
                 start: pint.Quantity,
                 end: Optional[pint.Quantity],
                 member: Any,
                 loadcase="D"):
        """
        Parameters
        ----------
        load_func : Callable or sp.Expr
            SymPy expression or callable that defines the load as a function of x
        start : pint.Quantity
            Starting position of load from left end
        end : pint.Quantity, optional
            End position of load (if None, assumed to be point load at start)
        member : Element Class
            The member that the load is applied to
        loadcase : str, optional
            Load case identifier
        """
        # Call parent initializer
        super().__init__(start, member, loadcase)

        self.load_func = load_func
        self.start = start
        self.end = end if end is not None else start
        self.kind = "SYMPY_LOAD"

        # Create symbolic variables
        x = sp.Symbol('x')
        E, I = sp.symbols('E I')

        # Convert function to SymPy expression if it's a callable
        if callable(load_func):
            # Sample points and create interpolating function
            print("Converting callable to SymPy expression")
            sample_points = [float(start.magnitude + i*(end.magnitude-start.magnitude)/20)
                           for i in range(21)]
            sample_values = [load_func(xi) for xi in sample_points]

            # Convert to piecewise polynomial using SymPy's interpolation
            self.sympy_expr = sp.interpolate(sample_points, sample_values, x)
        else:
            # Use the provided SymPy expression directly
            self.sympy_expr = load_func

        print(f"SymPy load expression: {self.sympy_expr}")

        # Create a SymPy beam
        L = float(self.L.magnitude)
        self.beam = Beam(L, E, I)

        # Apply the load to the beam
        if self.start == self.end:
            # Point load
            self.beam.apply_load(self.sympy_expr, float(self.start.magnitude), -1)
        else:
            # Distributed load
            self.beam.apply_load(self.sympy_expr, float(self.start.magnitude), 0,
                                end=float(self.end.magnitude))

        # Calculate reactions, shear, moment, slope and deflection using SymPy
        self._calculate_beam_responses()

        # Convert to PiecewisePolynomials for compatibility with the rest of the system
        self._convert_to_piecewise_polynomials()

    def _calculate_beam_responses(self):
        """Calculate beam responses using SymPy"""
        # Apply simple supports at ends for calculating the shear and moment
        self.beam.bc_deflection = [(0, 0), (float(self.L.magnitude), 0)]

        # Get the responses
        self.sympy_load = self.beam.load
        self.sympy_shear = self.beam.shear_force()
        self.sympy_moment = self.beam.bending_moment()
        self.sympy_slope = self.beam.slope()
        self.sympy_deflection = self.beam.deflection()

        print(f"SymPy load: {self.sympy_load}")
        print(f"SymPy shear: {self.sympy_shear}")
        print(f"SymPy moment: {self.sympy_moment}")

    def _convert_to_piecewise_polynomials(self):
        """Convert SymPy expressions to PiecewisePolynomials"""
        # TODO: This would require converting SymPy piecewise expressions
        # to your PiecewisePolynomial format

        # Example conversion (simplified):
        x = sp.Symbol('x')
        E, I = sp.symbols('E I')

        # For now, we'll implement a basic conversion by sampling points
        # This would need to be expanded to properly handle piecewise functions

        # Evaluate at sample points and create polynomial approximations
        self.Vy = self._sympy_to_ppoly(self.sympy_shear, x)
        self.Mz = self._sympy_to_ppoly(self.sympy_moment, x)

        # For slope and deflection, handle the EI factor
        slope_expr = self.sympy_slope * (self.E * self.I)
        defl_expr = self.sympy_deflection * (self.E * self.I)

        self.Sz = self._sympy_to_ppoly(slope_expr, x)
        self.Dy = self._sympy_to_ppoly(defl_expr, x)

    def _sympy_to_ppoly(self, sympy_expr, x_var):
        """Convert a SymPy expression to PiecewisePolynomial"""
        # This is a simplified conversion - would need enhancement for production

        try:
            # Attempt to convert directly if it's a polynomial
            if isinstance(sympy_expr, sp.Poly):
                coeffs = sympy_expr.all_coeffs()
                return PiecewisePolynomial([[coeffs, [0, self.L]]])

            # If it's a Piecewise expression
            if isinstance(sympy_expr, sp.Piecewise):
                pieces = []
                for expr, cond in sympy_expr.args:
                    # Extract region bounds from condition
                    # This is simplified and would need enhancement
                    if ">" in str(cond) and "<" in str(cond):
                        # Try to extract bounds
                        bounds_str = str(cond).split("&")
                        lower = float(bounds_str[0].split(">")[1])
                        upper = float(bounds_str[1].split("<")[1])

                        # Convert expression to polynomial coefficients
                        poly_expr = sp.Poly(expr, x_var)
                        coeffs = poly_expr.all_coeffs()

                        pieces.append([coeffs, [lower, upper]])

                return PiecewisePolynomial(pieces)

            # If conversion fails, use sampling approach
            print("Using sampling approach for SymPy to PiecewisePolynomial conversion")
            return self._sample_and_convert(sympy_expr, x_var)

        except Exception as e:
            print(f"Error converting SymPy expression to PiecewisePolynomial: {e}")
            # Fall back to sampling approach
            return self._sample_and_convert(sympy_expr, x_var)

    def _sample_and_convert(self, sympy_expr, x_var):
        """Sample the SymPy expression and create a PiecewisePolynomial"""
        from numpy.polynomial.polynomial import Polynomial
        import numpy as np

        # Sample the expression
        start_val = float(self.start.magnitude)
        end_val = float(self.end.magnitude)
        L_val = float(self.L.magnitude)

        # Create two regions: before and after the load
        regions = []

        if start_val > 0:
            # Region before the load
            x_points = np.linspace(0, start_val, 10)
            y_points = [float(sympy_expr.subs(x_var, xi)) for xi in x_points]
            poly = Polynomial.fit(x_points, y_points, 3)  # Cubic fit
            regions.append([list(poly.coef), [0, start_val]])

        # Region of the load
        x_points = np.linspace(start_val, end_val, 20)
        y_points = [float(sympy_expr.subs(x_var, xi)) for xi in x_points]
        poly = Polynomial.fit(x_points, y_points, 5)  # Higher degree for the load region
        regions.append([list(poly.coef), [start_val, end_val]])

        if end_val < L_val:
            # Region after the load
            x_points = np.linspace(end_val, L_val, 10)
            y_points = [float(sympy_expr.subs(x_var, xi)) for xi in x_points]
            poly = Polynomial.fit(x_points, y_points, 3)  # Cubic fit
            regions.append([list(poly.coef), [end_val, L_val]])

        return PiecewisePolynomial(regions)

    def FEF(self):
        """Compute and return the fixed end forces using SymPy"""
        # This would need to be implemented based on the SymPy expressions
        # Here's a skeleton implementation

        # Create a temporary beam with fixed ends
        x = sp.Symbol('x')
        E, I = sp.symbols('E I')
        L = float(self.L.magnitude)

        temp_beam = Beam(L, E, I)

        # Apply the load
        if self.start == self.end:
            # Point load
            temp_beam.apply_load(self.sympy_expr, float(self.start.magnitude), -1)
        else:
            # Distributed load
            temp_beam.apply_load(self.sympy_expr, float(self.start.magnitude), 0,
                                end=float(self.end.magnitude))

        # Set fixed end boundary conditions
        temp_beam.bc_deflection = [(0, 0), (L, 0)]
        temp_beam.bc_slope = [(0, 0), (L, 0)]

        # Get end forces and moments
        reactions = temp_beam.reaction_loads

        # Convert to the expected format
        # Assuming the reactions have the right units
        from pyMAOS import unit_manager

        # This will need adjustment based on how reaction_loads are structured
        Riy = unit_manager.ureg.Quantity(float(reactions.get('R_0', 0)),
                                       unit_manager.INTERNAL_FORCE_UNIT)
        Rjy = unit_manager.ureg.Quantity(float(reactions.get(f'R_{L}', 0)),
                                       unit_manager.INTERNAL_FORCE_UNIT)
        Miz = unit_manager.ureg.Quantity(float(reactions.get('M_0', 0)),
                                       unit_manager.INTERNAL_MOMENT_UNIT)
        Mjz = unit_manager.ureg.Quantity(float(reactions.get(f'M_{L}', 0)),
                                       unit_manager.INTERNAL_MOMENT_UNIT)

        # Create zeros with appropriate units
        zero_force = unit_manager.ureg.Quantity(0, unit_manager.INTERNAL_FORCE_UNIT)

        return [zero_force, Riy, Miz, zero_force, Rjy, Mjz]

    def __str__(self):
        """String representation of the SymPy load"""
        if self.start == self.end:
            return (f"SymPy Point Load ({self.loadcase}): "
                    f"expr={self.sympy_expr} at x={self.start:.3f}")
        else:
            return (f"SymPy Distributed Load ({self.loadcase}): "
                    f"expr={self.sympy_expr} from x={self.start:.3f} to x={self.end:.3f}")

if __file__ == "main":
    # Example using a SymPy expression for a triangular load
    from pyMAOS import unit_manager
    import sympy as sp

    # Define symbolic variable
    x = sp.Symbol('x')

    # Create a member
    L = unit_manager.ureg.Quantity(10, 'm')
    member = SomeBeamClass(L, material, section)  # Replace with your actual beam class

    # Define a triangular load that peaks at the center
    # q(x) = q0 * (1 - |2x/L - 1|)
    q0 = unit_manager.ureg.Quantity(5, 'kN/m')
    load_expr = q0.magnitude * (1 - sp.Abs(2*x/L.magnitude - 1))

    # Create the SymPy load
    load = R2_SymPy_Load(load_expr,
                         start=unit_manager.ureg.Quantity(0, 'm'),
                         end=L,
                         member=member)

    # Analyze the load
    load.print_detailed_analysis()