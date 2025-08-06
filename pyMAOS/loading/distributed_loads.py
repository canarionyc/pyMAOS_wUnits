import sys
import pint
from pint import Quantity
from typing import TYPE_CHECKING, Any
# from pyMAOS.display_utils import display_node_load_vector_in_units
import numpy as np
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial
import pyMAOS
from pyMAOS import unit_manager, INTERNAL_LENGTH_UNIT
from pyMAOS import (# SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS,
    INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT,  INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT,
    FORCE_DIMENSIONALITY, LENGTH_DIMENSIONALITY, MOMENT_DIMENSIONALITY, PRESSURE_DIMENSIONALITY, DISTRIBUTED_LOAD_DIMENSIONALITY,
    unit_manager
)

from pprint import pprint

# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from pyMAOS.frame2d import R2Frame

class LinearLoadXY:
    def __init__(self, w1: pint.Quantity, w2: pint.Quantity, a: pint.Quantity, b: pint.Quantity, member: "Any", loadcase="D"):
        self.w1 = w1
        self.w2 = w2
        self.a = a
        self.b = b
        self.c = b - a
        self.member_uid = member.uid
        self.L = member.length

        EI = member.material.E *  member.section.Ixx
        EI = EI.to_reduced_units()
        self.kind = "LINE"
        self.loadcase = loadcase

        # Constants of Integration
        # self.integration_constants()
        """
                Calculate the integration constants for beam deflection equations.

                These constants (c01-c12) are determined by enforcing boundary conditions:
                - Continuity of shear, moment, slope, and deflection at load discontinuities
                - Zero displacement and rotation at member ends (for fixed-end conditions)
                - Equilibrium of forces and moments
                """
        # w1 = self.w1
        # w2 = self.w2
        # a = self.a
        # b = self.b
        L = self.L

        #     These constants (c01-c12) are determined by enforcing boundary conditions:
        #     - Continuity of shear, moment, slope, and deflection at load discontinuities
        #     - Zero displacement and rotation at member ends (for fixed-end conditions)
        #     - Equilibrium of forces and moments
        #
        #     The constants are used in the piecewise polynomial functions that define:
        #     - Vy: Shear force distribution
        #     - Mz: Bending moment distribution
        #     - Sz: Slope (rotation) distribution
        #     - Dy: Deflection distribution

        # Constants for the shear force function (Vy)
        self.c01 = (
                          (((2 * b * b) + ((-a - 3 * L) * b) - (a * a) + (3 * L * a)) * w2)
                          + (((b * b) + ((a - 3 * L) * b) - (2 * a * a) + (3 * L * a)) * w1)
                  ) / (6 * L)
        self.c02 = (
                          (
                                  (
                                          (2 * b * b * b)
                                          + ((-3 * a - 3 * L) * b * b)
                                          + (6 * L * a * b)
                                          + (a * a * a)
                                  )
                                  * w2
                          )
                          + (((b * b * b) - (3 * L * b * b) - (3 * a * a * b) + (2 * a * a * a)) * w1)
                  ) / (6 * L * b - 6 * L * a)
        self.c03 = (
                          ((2 * b * b - a * b - a * a) * w2) + ((b * b + a * b - 2 * a * a) * w1)
                  ) / (6 * L)

        # Constants for the bending moment function (Mz)
        # Use unit_manager.ureg instead of direct import
        self.c04 = pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT)  # Zero moment at x=0 for fixed-end condition
        self.c05 = (
                -1
                * ((a * a * a * w2) + ((2 * a * a * a - 3 * a * a * b) * w1))
                / (6 * b - 6 * a)
        )
        self.c06 = (
                -1
                * ((2 * b * b - a * b - a * a) * w2 + (b * b + a * b - 2 * a * a) * w1)
                / 6
        )

        # Constants for the slope function (Sz)
        self.c07 = (
                          (
                                  12 * b * b * b * b
                                  + (-3 * a - 45 * L) * b * b * b
                                  + (-3 * a * a + 15 * L * a + 40 * L * L) * b * b
                                  + (-3 * a * a * a + 15 * L * a * a - 20 * L * L * a) * b
                                  - 3 * a * a * a * a
                                  + 15 * L * a * a * a
                                  - 20 * L * L * a * a
                          )
                          * w2
                          + (
                                  3 * b * b * b * b
                                  + (3 * a - 15 * L) * b * b * b
                                  + (3 * a * a - 15 * L * a + 20 * L * L) * b * b
                                  + (3 * a * a * a - 15 * L * a * a + 20 * L * L * a) * b
                                  - 12 * a * a * a * a
                                  + 45 * L * a * a * a
                                  - 40 * L * L * a * a
                          )
                          * w1
                  ) / (360 * L)
        self.c08 = (
                          (
                                  12 * b * b * b * b * b
                                  + (-15 * a - 45 * L) * b * b * b * b
                                  + (60 * L * a + 40 * L * L) * b * b * b
                                  - 60 * L * L * a * b * b
                                  + 3 * a * a * a * a * a
                                  + 20 * L * L * a * a * a
                          )
                          * w2
                          + (
                                  3 * b * b * b * b * b
                                  - 15 * L * b * b * b * b
                                  + 20 * L * L * b * b * b
                                  + (-15 * a * a * a * a - 60 * L * L * a * a) * b
                                  + 12 * a * a * a * a * a
                                  + 40 * L * L * a * a * a
                          )
                          * w1
                  ) / (360 * L * b - 360 * L * a)
        self.c09 = (
                          (
                                  12 * b * b * b * b
                                  - 3 * a * b * b * b
                                  + (40 * L * L - 3 * a * a) * b * b
                                  + (-3 * a * a * a - 20 * L * L * a) * b
                                  - 3 * a * a * a * a
                                  - 20 * L * L * a * a
                          )
                          * w2
                          + (
                                  3 * b * b * b * b
                                  + 3 * a * b * b * b
                                  + (3 * a * a + 20 * L * L) * b * b
                                  + (3 * a * a * a + 20 * L * L * a) * b
                                  - 12 * a * a * a * a
                                  - 40 * L * L * a * a
                          )
                          * w1
                  ) / (360 * L)

        # Constants for the deflection function (Dy)
        self.c10 = pyMAOS.unit_manager.ureg.Quantity(0, f"{pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT}**3 * {pyMAOS.unit_manager.INTERNAL_FORCE_UNIT}")  # Zero deflection at x=0 for fixed-end condition
        self.c11 = (
                -1 /120
                * (
                        a * a * a * a * a * w2
                        + (4 * a * a * a * a * a - 5 * a * a * a * a * b) * w1
                )
                / (b - a)
        )
        self.c12 = (
                -1
                * (
                        (
                                4 * b * b * b * b
                                - a * b * b * b
                                - a * a * b * b
                                - a * a * a * b
                                - a * a * a * a
                        )
                        * w2
                        + (
                                b * b * b * b
                                + a * b * b * b
                                + a * a * b * b
                                + a * a * a * b
                                - 4 * a * a * a * a
                        )
                        * w1
                )
                / 120
        )

        # Simple End Reactions
        self.W = 0.5 * self.c * (self.w2 + self.w1)
        self.cbar = ((self.w1 + (2 * self.w2)) / (3 * (self.w2 + self.w1))) * self.c

        self.Rjy = -1 * self.W * (self.a + self.cbar) * (1 / self.L)
        self.Riy = -1 * self.W - self.Rjy

        # Piecewise Functions
        # Each piecewise function represents a different structural response:
        # - Wy: Distributed load function (input)
        # - Vy: Shear force distribution (integral of Wy)
        # - Mz: Bending moment distribution (integral of Vy)
        # - Sz: Rotation/slope distribution (integral of Mz/EI)
        # - Dy: Deflection distribution (integral of Sz)
        #
        # Each function is defined in three pieces:
        # 1. Before loaded region [0 to a]
        # 2. Within loaded region [a to b] 
        # 3. After loaded region [b to L]
        #
        # Format: [[coefficients], [domain_bounds]]
        # where coefficients = [c₀, c₁, c₂...] representing c₀ + c₁x + c₂x² + ...
        Wy = [
            [[pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_DISTRIBUTED_LOAD_UNIT)], [pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT), self.a]],
            [
                [
                    ((-1 * self.a * self.w2) - (self.c * self.w1) - (self.a * self.w1))
                    / self.c,
                    (self.w2 - self.w1) / self.c,
                ],
                [self.a, self.b],
            ],
            [[pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_DISTRIBUTED_LOAD_UNIT)], [self.b, self.L]],
        ]; print("Wy:\n", Wy)

        Vy = [
            [[self.c01], [unit_manager.ureg.Quantity(0, INTERNAL_LENGTH_UNIT), self.a]],
            [
                [
                    self.c02,
                    self.w1
                    + ((self.a * self.w1) / self.c)
                    - ((self.a * self.w2) / self.c),
                    (self.w2 / (2 * self.c)) - (self.w1 / (2 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c03], [self.b, self.L]],
        ]; print("Vy:\n", Vy)

        Mz = [
            [[self.c04, self.c01], [unit_manager.ureg.Quantity(0, INTERNAL_LENGTH_UNIT), self.a]],
            [
                [
                    self.c05,
                    self.c02,
                    (self.w1 / 2)
                    + ((self.a * self.w1) / (2 * self.c))
                    - ((self.a * self.w2) / (2 * self.c)),
                    (self.w2 / (6 * self.c)) - (self.w1 / (6 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c06, self.c03], [self.b, self.L]],
        ]; print("Mz:\n", Mz)

        Sz = [
            [[self.c07, self.c04, 0.5 * self.c01], [unit_manager.ureg.Quantity(0, INTERNAL_LENGTH_UNIT), self.a]],
            [
                [
                    self.c08,
                    self.c05,
                    0.5 * self.c02,
                    (self.w1 / 6)
                    + ((self.a * self.w1) / (6 * self.c))
                    - ((self.a * self.w2) / (6 * self.c)),
                    (self.w2 / (24 * self.c)) - (self.w1 / (24 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c09, self.c06, 0.5 * self.c03], [self.b, self.L]],
        ]
        Sz[0][0] = [i / EI for i in Sz[0][0]]
        Sz[1][0] = [i / EI for i in Sz[1][0]]
        Sz[2][0] = [i / EI for i in Sz[2][0]]

        print("Sz:\n", Sz)

        Dy = [
            [[self.c10, self.c07, 0.5 * self.c04, self.c01 / 6], [unit_manager.ureg.Quantity(0, INTERNAL_LENGTH_UNIT), self.a]],
            [
                [
                    self.c11,
                    self.c08,
                    0.5 * self.c05,
                    self.c02 / 6,
                    (self.w1 / 24)
                    + ((self.a * self.w1) / (24 * self.c))
                    - ((self.a * self.w2) / (24 * self.c)),
                    (self.w2 / (120 * self.c)) - (self.w1 / (120 * self.c)),
                ],
                [self.a, self.b],
            ],
            [
                [self.c12, self.c09, 0.5 * self.c06, self.c03 / 6],
                [self.b, self.L],
            ],
        ]

        Dy[0][0] = [i / EI for i in Dy[0][0]]
        Dy[1][0] = [i / EI for i in Dy[1][0]]
        Dy[2][0] = [i / EI for i in Dy[2][0]]
        print("Dy:\n", Dy)
        import inspect
        print(f"{inspect.getfile(inspect.currentframe())}:{inspect.currentframe().f_lineno}")
        print("Dy:", Dy, sep="\n")
        from display_utils import print_quantity_nested_list; print_quantity_nested_list(Dy,simplify_units=True)
        # self.Wx = PiecewisePolynomial()  # Axial Load Function
        self.Wy = PiecewisePolynomial(Wy); print("Wy:", self.Wy, sep="\n") # Vertical Load Function

        self.Ax = PiecewisePolynomial()
        self.Dx = PiecewisePolynomial()
        from pprint import pprint; pprint(Vy); self.Vy = PiecewisePolynomial(Vy); print("Vy:", self.Vy, sep="\n")
        print("Mz="); pprint(Mz, width=240); self.Mz = PiecewisePolynomial(Mz); print("Mz:", self.Mz, sep="\n") # this is a moment
        print("Sz="); pprint(Sz, width=240); self.Sz = PiecewisePolynomial(Sz); print("Sz:", self.Sz,sep="\n") # this is an angle
        print("Dy="); pprint(Dy, width=240); self.Dy = PiecewisePolynomial(Dy); print("Dy:", self.Dy,sep="\n")

        # After creating all polynomial objects
        # fig = self.plot_all_functions()
        # fig.show()  # If you want to display immediately

        from PiecewisePolynomial2 import PiecewisePolynomial2
        # Create PiecewisePolynomial2 objects for each function
        self.Wy2 = PiecewisePolynomial2(Wy)
        self.Vy2 = PiecewisePolynomial2(Vy)
        self.Mz2 = PiecewisePolynomial2(Mz)
        self.Sz2 = PiecewisePolynomial2(Sz)
        self.Dy2 = PiecewisePolynomial2(Dy)
        # Print the PiecewisePolynomial2 objects
        print("Wy2:", self.Wy2)
        print("Vy2:", self.Vy2)
        print("Mz2:", self.Mz2)
        print("Sz2:", self.Sz2)
        print("Dy2:", self.Dy2)

        # After creating PiecewisePolynomial2 objects
        ppoly_fig = self.plot_all_ppoly_functions()
        ppoly_fig.show()  # If you want to display immediately


    def load_fef(self):
        """
        Compute and return the fixed and forces
        """
        L = self.L

        c3 = self.c03
        c6 = self.c06
        c7 = self.c07
        c9 = self.c09

        # Calculate fixed end moments
        Miz = -1 * (c3 * L * L + 2 * c6 * L + 2 * c9 + 4 * c7) / L
        Mjz = -1 * (2 * c3 * L * L + 4 * c6 * L + 4 * c9 + 2 * c7) / L

        # Calculate fixed end forces
        Riy = self.Riy + (Miz / L) + (Mjz / L)
        Rjy = self.Rjy - (Miz / L) - (Mjz / L)

        # Dimension checking
        # print(f"DEBUG: Checking dimensions - Miz: {Miz.dimensionality}, Mjz: {Mjz.dimensionality}")
        # print(f"DEBUG: Checking dimensions - Riy: {Riy.dimensionality}, Rjy: {Rjy.dimensionality}")

        # Verify moment dimensions
        # try:
        #     Miz.check(MOMENT_DIMENSIONALITY)
        #     Mjz.check(MOMENT_DIMENSIONALITY)
        #     # print("DEBUG: Moment dimension check passed")
        # except pint.DimensionalityError as e:
        #     print(f"ERROR: Dimension error in moments: {e}")
        #     # Create correctly dimensioned values as fallback
        #     if not Miz.check(MOMENT_DIMENSIONALITY):
        #         print(f"WARNING: Fixing dimensions of Miz from {Miz.dimensionality} to {MOMENT_DIMENSIONALITY}")
        #         Miz = unit_manager.ureg.Quantity(Miz.magnitude, INTERNAL_MOMENT_UNIT)
        #     if not Mjz.check(MOMENT_DIMENSIONALITY):
        #         print(f"WARNING: Fixing dimensions of Mjz from {Mjz.dimensionality} to {MOMENT_DIMENSIONALITY}")
        #         Mjz = unit_manager.ureg.Quantity(Mjz.magnitude, INTERNAL_MOMENT_UNIT)

        # Verify force dimensions
        # try:
        #     Riy.check(FORCE_DIMENSIONALITY)
        #     Rjy.check(FORCE_DIMENSIONALITY)
        #     # print("DEBUG: Force dimension check passed")
        # except pint.DimensionalityError as e:
        #     print(f"ERROR: Dimension error in forces: {e}")
        #     # Create correctly dimensioned values as fallback
        #     if not Riy.check(FORCE_DIMENSIONALITY):
        #         print(f"WARNING: Fixing dimensions of Riy from {Riy.dimensionality} to {FORCE_DIMENSIONALITY}")
        #         Riy = unit_manager.ureg.Quantity(Riy.magnitude, INTERNAL_FORCE_UNIT)
        #     if not Rjy.check(FORCE_DIMENSIONALITY):
        #         print(f"WARNING: Fixing dimensions of Rjy from {Rjy.dimensionality} to {FORCE_DIMENSIONALITY}")
        #         Rjy = unit_manager.ureg.Quantity(Rjy.magnitude, INTERNAL_FORCE_UNIT)

        # Print forces and moments in both SI and display units
        #from pyMAOS.units_mod import convert_to_display_units
        from pyMAOS.pymaos_units import FORCE_DISPLAY_UNIT, MOMENT_DISPLAY_UNIT
        # Get current unit system directly from the manager
        # current_units = unit_manager.get_current_units()
        # system_name = unit_manager.get_system_name()
        # Riy_display = Riy.to(FORCE_DISPLAY_UNIT)
        # Rjy_display = Rjy.to(FORCE_DISPLAY_UNIT)
        # Miz_display = Miz.to(MOMENT_DISPLAY_UNIT)
        # Mjz_display = Mjz.to(MOMENT_DISPLAY_UNIT)
        #
        # print(f"Vertical reactions - SI: Riy={Riy:.3f} N, Rjy={Rjy:.3f} N")
        # print(f"Vertical reactions - Display: Riy={Riy_display:.3f}, Rjy={Rjy_display:.3f}")
        # print(f"Moments - SI: Miz={Miz:.3f} N*m, Mjz={Mjz:.3f} N*m")
        # print(f"Moments - Display: Miz={Miz_display:.3f}, Mjz={Mjz_display:.3f}")

        # Create zeros with appropriate units
        zero_force = 0 * pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT)

        ret_val = np.array([pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT), Riy, Miz, pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT), Rjy, Mjz], dtype=object)
        print(f"FEF distributed load results on member {self.member_uid} for Load Case {self.loadcase}:", ret_val, sep="\n")

        # Final dimension check for return values
        for i, (idx, expected_dim) in enumerate([(0, FORCE_DIMENSIONALITY), (1, FORCE_DIMENSIONALITY),
                                               (2, MOMENT_DIMENSIONALITY), (3, FORCE_DIMENSIONALITY),
                                               (4, FORCE_DIMENSIONALITY), (5, MOMENT_DIMENSIONALITY)]):
            try:
                ret_val[idx].check(expected_dim)
            except pint.DimensionalityError as e:
                print(f"Dimensionality error in ret_val[{idx}]: {e}")
                print(f"  Actual: {ret_val[idx].dimensionality}, Expected: {expected_dim}")

        return ret_val

    def __str__(self):
        """
        String representation of a linear load.
        
        Returns:
        -------
        str
            Description of the linear load including magnitude, position, and load case.
        """
        return (f"Linear Load ({self.loadcase}): "
                f"w1={self.w1:.3f}, w2={self.w2:.3f}, "
                f"from x={self.a:.3f} to x={self.b:.3f} "
                f"(on member of length {self.L:.3f})")

    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Prints detailed analysis of beam response across all three regions with ASCII charts.

        Parameters
        ----------
        num_points : int
            Number of points to sample in each region
        chart_width : int
            Width of ASCII charts in characters
        chart_height : int
            Height of ASCII charts in characters
        """

        
        import numpy as np

        # Get current unit system directly from the manager
        current_units = unit_manager.get_current_units()
        system_name = unit_manager.get_system_name()
        # print(f"\n===== DETAILED ANALYSIS FOR {self.__str__()} =====")
        # print(f"Total Load W = {self.W:.3f} {INTERNAL_FORCE_UNIT} ({convert_to_display_units(self.W, 'force'):.3f} {current_units['force']})")
        # print(f"Load centroid from left: {self.a + self.cbar:.3f}")
        # print(f"Reactions: Riy = {self.Riy:.3f} {INTERNAL_FORCE_UNIT} ({convert_to_display_units(self.Riy, 'force'):.3f} {current_units['force']}), Rjy = {self.Rjy:.3f} {INTERNAL_FORCE_UNIT} ({convert_to_display_units(self.Rjy, 'force'):.3f} {current_units['force']})", end="\n")

        # Sample points across all regions
        regions = [(0, self.a), (self.a, self.b), (self.b, self.L)]
        region_names = ["Before Load [0 to a]", "Loaded Region [a to b]", "After Load [b to L]"]

        # Create sampling points
        all_x = []
        for i, (start, end) in enumerate(regions):
            if end > start:  # Only if region has non-zero width
                points = [start + j*(end-start)/num_points for j in range(num_points+1)]
                # points=np.linspace(start, end, num_points+1).tolist()
                print(f"Region {i+1} ({region_names[i]}): {points}")
                # Don't duplicate boundary points
                if i > 0 and len(all_x) > 0:
                    points = points[1:]
                all_x.extend(points)

        # Convert to numpy array
        x_array = np.array(all_x, dtype=object)

        # Calculate function values using vectorized evaluation
        wy_values = self.Wy.evaluate_vectorized(x_array)
        vy_values = self.Vy.evaluate_vectorized(x_array)
        mz_values = self.Mz.evaluate_vectorized(x_array)
        sz_values = self.Sz.evaluate_vectorized(x_array)
        dy_values = self.Dy.evaluate_vectorized(x_array)

        # Print ASCII charts
        self._print_ascii_chart("Distributed Load (Wy)", all_x, wy_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Shear Force (Vy)", all_x, vy_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Bending Moment (Mz)", all_x, mz_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Rotation (Sz)", all_x, sz_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Deflection (Dy)", all_x, dy_values, regions, chart_width, chart_height)

        # Print table of values at region boundaries
        print("\n===== VALUES AT KEY POINTS =====")
        print(f"{'Position':15} {'Load':15} {'Shear':15} {'Moment':15} {'Rotation':15} {'Deflection':15}")
        print("-" * 90)
        for x in [0, self.a, self.b, self.L]:
            print(f"{x:15.3f} {self.Wy.evaluate(x):15.3f} {self.Vy.evaluate(x):15.3f} {self.Mz.evaluate(x):15.3f} "
                  f"{self.Sz.evaluate(x):15.3e} {self.Dy.evaluate(x):15.3e}")
    
    def _print_ascii_chart(self, title, x_values, y_values, regions, width=60, height=15):
        """
        Helper method to print an ASCII chart of data with proper unit handling.
        """
        import numpy as np
        from pint import Quantity  # Import Quantity in the method scope

        if len(y_values) == 0:
            return

        print(f"\n--- {title} ---")

        # Filter out NaN values before finding min/max
        valid_indices = []
        valid_y_values = []
        for i, y in enumerate(y_values):
            # Check if y is NaN (including Quantity objects with NaN magnitude)
            is_nan = False
            if hasattr(y, 'magnitude'):
                is_nan = np.isnan(y.magnitude)
            else:
                is_nan = np.isnan(y) if isinstance(y, (int, float)) else False

            if not is_nan:
                valid_indices.append(i)
                valid_y_values.append(y)

        # If no valid values, skip plotting
        if len(valid_y_values) == 0:
            print("No valid data points to plot (all values are NaN)")
            return

        # Find min and max values while preserving units
        min_y = min(valid_y_values)
        max_y = max(valid_y_values)

        # Debug print
        print(f"Value range: {min_y:.3f} to {max_y:.3f}")

        # Avoid division by zero
        if min_y == max_y:
            if hasattr(min_y, 'magnitude') and min_y.magnitude == 0:
                # Create non-zero range with proper units
                if hasattr(min_y, 'units'):
                    min_y -= 1 * min_y.units
                    max_y += 1 * max_y.units
                else:
                    min_y -= 1
                    max_y += 1
            else:
                # Just create some range around the value
                min_y = 0.9 * min_y
                max_y = 1.1 * max_y

        # Get the maximum x value for scaling
        max_x = max(x_values)
        range_y = max_y - min_y

        # Create the chart grid
        chart = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw x-axis if zero is in the range
        if min_y <= 0 <= max_y:
            # Calculate position while preserving units
            axis_pos = height - int(height * (0 - min_y) / range_y)
            axis_pos = max(0, min(height - 1, axis_pos))
            chart[axis_pos] = ['-' for _ in range(width)]

        # Plot data points
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            # Skip NaN values
            if hasattr(y, 'magnitude'):
                if np.isnan(y.magnitude):
                    continue
            elif isinstance(y, (int, float)) and np.isnan(y):
                continue

            # Map x and y to chart coordinates while preserving units
            x_pos = int(width * x / max_x)
            x_pos = min(width - 1, max(0, x_pos))

            # Calculate y position in chart - avoid NaN issues
            try:
                y_normalized = (y - min_y) / range_y
                y_pos = height - 1 - int(y_normalized * (height - 1))
                y_pos = min(height - 1, max(0, y_pos))
                chart[y_pos][x_pos] = '*'
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"Warning: Could not plot point at x={x}, y={y}: {e}")
                continue

        # Draw vertical lines at region boundaries
        for start, end in regions:
            for boundary in [start, end]:
                if boundary > 0 and boundary < max_x:
                    x_pos = int(width * boundary / max_x)
                    x_pos = min(width - 1, max(0, x_pos))
                    for y_pos in range(height):
                        if chart[y_pos][x_pos] != '*':  # Don't overwrite data points
                            chart[y_pos][x_pos] = '|'

        # Print the chart
        for row in chart:
            print(''.join(row))

        # Print region information
        print(f"Region boundaries: [{unit_manager.ureg.Quantity(0, self.a.units)}, {self.a:.2f}, {self.b:.2f}, {self.L:.2f}]")

    def plot_all_functions(self, figsize=(10, 12), convert_x_to=None, convert_y_to=None):
        """
        Create a figure with subplots for all non-empty PiecewisePolynomial functions.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        convert_x_to : pint.Unit, optional
            Convert x values to this unit for plotting
        convert_y_to : dict, optional
            Dictionary mapping function name to unit for conversion, e.g. {'Vy': 'kN'}

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing all plots
        """
        import matplotlib.pyplot as plt

        # Collect all non-empty PiecewisePolynomial objects with their names and colors
        functions = []
        if hasattr(self, 'Wy') and self.Wy.functions:
            functions.append(('Wy', self.Wy, 'blue', 'Distributed Load'))
        if hasattr(self, 'Vy') and self.Vy.functions:
            functions.append(('Vy', self.Vy, 'red', 'Shear Force'))
        if hasattr(self, 'Mz') and self.Mz.functions:
            functions.append(('Mz', self.Mz, 'green', 'Bending Moment'))
        if hasattr(self, 'Sz') and self.Sz.functions:
            functions.append(('Sz', self.Sz, 'purple', 'Rotation'))
        if hasattr(self, 'Dy') and self.Dy.functions:
            functions.append(('Dy', self.Dy, 'orange', 'Deflection'))

        # Return early if no functions to plot
        if not functions:
            print("No functions to plot")
            return None

        # Create figure and subplots
        fig, axes = plt.subplots(len(functions), 1, figsize=figsize, sharex=True)

        # Handle single subplot case
        if len(functions) == 1:
            axes = [axes]

        print(f"Plotting {len(functions)} functions")

        # Create each plot
        for i, (name, func, color, title) in enumerate(functions):
            # Convert y units if specified
            y_unit = None
            if convert_y_to and name in convert_y_to:
                y_unit = convert_y_to[name]

            # Plot the function on the appropriate subplot
            func.plot(
                ax=axes[i],
                color=color,
                title=f"{title} ({name})",
                convert_x_to=convert_x_to,
                convert_y_to=y_unit,
                show=False
            )

            # Add vertical lines at key points
            for x in [self.a, self.b]:
                if hasattr(x, 'magnitude'):
                    x_val = x.to(convert_x_to).magnitude if convert_x_to else x.magnitude
                else:
                    x_val = x
                axes[i].axvline(x=x_val, color='gray', linestyle='--', alpha=0.7)

        # Add overall title
        fig.suptitle(f"Beam Analysis for {self.__str__()}", fontsize=16)

        # Adjust spacing
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        # Show the grid on all plots
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.7)

        return fig

    def plot_all_ppoly_functions(self, figsize=(10, 12), convert_x_to=None, convert_y_to=None):
        """
        Create a figure with subplots for all PiecewisePolynomial2 functions.
        Uses the efficient PPoly representation for better performance.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        convert_x_to : pint.Unit, optional
            Convert x values to this unit for plotting
        convert_y_to : dict, optional
            Dictionary mapping function name to unit for conversion, e.g. {'Vy': 'kN'}

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing all plots
        """
        import matplotlib.pyplot as plt

        # Collect all non-empty PiecewisePolynomial2 objects
        functions = []
        if hasattr(self, 'Wy2') and self.Wy2.ppoly is not None:
            functions.append(('Wy2', self.Wy2, 'blue', 'Distributed Load'))
        if hasattr(self, 'Vy2') and self.Vy2.ppoly is not None:
            functions.append(('Vy2', self.Vy2, 'red', 'Shear Force'))
        if hasattr(self, 'Mz2') and self.Mz2.ppoly is not None:
            functions.append(('Mz2', self.Mz2, 'green', 'Bending Moment'))
        if hasattr(self, 'Sz2') and self.Sz2.ppoly is not None:
            functions.append(('Sz2', self.Sz2, 'purple', 'Rotation'))
        if hasattr(self, 'Dy2') and self.Dy2.ppoly is not None:
            functions.append(('Dy2', self.Dy2, 'orange', 'Deflection'))

        # Return early if no functions to plot
        if not functions:
            print("No PiecewisePolynomial2 functions to plot")
            return None

        # Create figure and subplots
        fig, axes = plt.subplots(len(functions), 1, figsize=figsize, sharex=True)

        # Handle single subplot case
        if len(functions) == 1:
            axes = [axes]

        print(f"Plotting {len(functions)} PiecewisePolynomial2 functions")

        # Create each plot
        for i, (name, func, color, title) in enumerate(functions):
            # Convert y units if specified
            y_unit = None
            if convert_y_to and name in convert_y_to:
                y_unit = convert_y_to[name]

            # Plot the function on the appropriate subplot
            func.plot(
                ax=axes[i],
                color=color,
                title=f"{title} ({name})",
                convert_x_to=convert_x_to,
                convert_y_to=y_unit,
                show=False
            )

            # Add vertical lines at key load boundaries
            for x in [self.a, self.b]:
                if hasattr(x, 'magnitude'):
                    x_val = x.to(convert_x_to).magnitude if convert_x_to else x.magnitude
                else:
                    x_val = x
                axes[i].axvline(x=x_val, color='gray', linestyle='--', alpha=0.7)

        # Add overall title
        fig.suptitle(f"Beam Analysis (PPoly) for {self.__str__()}", fontsize=16)

        # Adjust spacing
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        # Show the grid on all plots
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.7)

        return fig