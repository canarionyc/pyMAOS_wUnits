import pint
from typing import TYPE_CHECKING, Any
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial
from pprint import pprint
from display_utils import print_quantity_nested_list
import pyMAOS
# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from pyMAOS.frame2d import R2Frame

class R2_Point_Moment:
    def __init__(self, M: pint.Quantity, a: pint.Quantity, member: "Any", loadcase="D"):
        """
        Parameters
        ----------
        M : FLOAT
            Applied moment, counter-clockwise positive.
        a : FLOAT
            Point of application of moment as measured from the member left end.
        member : Element Class
            the member that the load is applied to.
        loadcase : STRING, optional
            String representation of the applied load type, this
            data is used for load cases and combindations. The default is "D".
        """
        self.M = M
        self.a = a
        self.L = member.length

        self.E = member.material.E
        self.I = member.section.Ixx

        self.EI = self.E * self.I

        self.kind = "MOMENT"
        self.loadcase = loadcase

        # Constants of Integration
        self.integration_constants()

        # Simple End Reactions
        self.Riy = self.M / self.L
        self.Rjy = -1 * self.Riy

        # Piecewise Functions
        # [co....cn x^n] [xa, xb]

        Vy = [[[self.Riy], [0, self.a]], [[self.Riy], [self.a, self.L]]]
        pprint(Vy)
        Mz = [
            [[0, self.Riy], [0, self.a]],
            [[-1 * self.M, self.Riy], [self.a, self.L]],
        ]
        pprint(Mz)
        Sz = [
            [[self.c1 / self.EI, 0, self.Riy / (2 * self.EI)], [0, self.a]],
            [
                [
                    self.c2 / self.EI,
                    -1 * self.M / self.EI,
                    self.Riy / (2 * self.EI),
                ],
                [self.a, self.L],
            ],
        ]
        pprint(Sz)
        Dy = [
            [
                [
                    self.c3 / self.EI,
                    self.c1 / self.EI,
                    0,
                    self.Riy / (6 * self.EI),
                ],
                [0, self.a],
            ],
            [
                [
                    self.c4 / self.EI,
                    self.c2 / self.EI,
                    -1 * self.M / (2 * self.EI),
                    self.Riy / (6 * self.EI),
                ],
                [self.a, self.L],
            ],
        ]
        pprint(Dy)
        self.Wx = PiecewisePolynomial()  # Axial Load Function
        self.Wy = PiecewisePolynomial()  # Vertical Load Function
        self.Ax = PiecewisePolynomial()
        self.Dx = PiecewisePolynomial()
        self.Vy = PiecewisePolynomial(Vy)
        self.Mz = PiecewisePolynomial(Mz)
        self.Sz = PiecewisePolynomial(Sz)
        self.Dy = PiecewisePolynomial(Dy)

    def integration_constants(self):
        M = self.M
        a = self.a
        L = self.L

        # Constants of Integration
        self.c1 = ((3 * M * a * a) - (6 * L * M * a) + (2 * L * L * M)) / (6 * L)
        self.c2 = ((3 * M * a * a) + (2 * L * L * M)) / (6 * L)
        self.c3 = 0
        self.c4 = -1 / 2 * M * a * a

    def FEF(self):
        """
        Compute and return the fixed and forces
        """
        M = self.M
        a = self.a
        L = self.L

        Miz = -1 * (M * (a - L) * ((3 * a) - L)) / (L * L)
        Mjz = -1 * (M * a * (3 * a - 2 * L)) / (L * L)
        Riy = self.Riy + (Miz / L) + (Mjz / L)
        Rjy = self.Rjy - (Miz / L) - (Mjz / L)

        # Create zeros with appropriate units
        zero_force = 0 * unit_manager.ureg(Riy.units)

        return [zero_force, Riy, Miz, zero_force, Rjy, Mjz]


class R2_Point_Load:
    def __init__(self, p: pint.Quantity, a: pint.Quantity, member: "Any", loadcase="D"):
        self.p = p
        self.a = a
        self.L = member.length

        self.E = member.material.E
        self.I = member.section.Ixx

        self.EI = self.E * self.I

        self.kind = "POINT"
        self.loadcase = loadcase

        # Constants of Integration
        # self.integration_constants()
        # p = self.p
        # a = self.a
        L = self.L
        import inspect;
        print(f"{inspect.getfile(inspect.currentframe())}:{inspect.currentframe().f_lineno}")

        from pyMAOS import INTERNAL_LENGTH_UNIT,INTERNAL_FORCE_UNIT,INTERNAL_MOMENT_UNIT
        self.c1 = pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT)
        self.c2 = -1 * p * a
        self.c3 = (p * a * (a - (2 * L)) * (a - L)) / (6 * L)
        self.c4 = (p * a * ((a * a) + (2 * L * L))) / (6 * L)
        self.c5 = pyMAOS.unit_manager.ureg.Quantity(0, f"{pyMAOS.unit_manager.INTERNAL_FORCE_UNIT} * {pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT}**3")
        self.c6 = (-1 * p * a * a * a) / 6

        # Simple End Reactions
        self.Riy = self.p * ((self.a - self.L) / self.L)
        self.Rjy = -1 * self.p * self.a * (1 / self.L)
        print(f"Riy: {self.Riy:.3f}, Rjy: {self.Rjy:.3f}")
        # Piecewise Functions
        # [co....cn x^n] [xa, xb]
        Vy = [
            [[self.Riy], [pyMAOS.unit_manager.ureg.Quantity(0, self.a.units), self.a]],
            [[self.Riy + self.p], [self.a, self.L]],
        ]
        print("Vy:"); print_quantity_nested_list(Vy)
        Mz = [
            [[self.c1, self.Riy], [pyMAOS.unit_manager.ureg.Quantity(0, self.a.units), self.a]],
            [[self.c2, self.Riy + self.p], [self.a, self.L]],
        ]
        print("Mz:"); print_quantity_nested_list(Mz)
        Sz = [
            [[self.c3, self.c1, self.Riy / 2], [pyMAOS.unit_manager.ureg.Quantity(0, self.a.units), self.a]],
            [[self.c4, self.c2, (self.Riy + self.p) / 2], [self.a, self.L]],
        ]
        Sz[0][0] = [i / self.EI for i in Sz[0][0]]
        Sz[1][0] = [i / self.EI for i in Sz[1][0]]
        print("Sz:"); print_quantity_nested_list(Sz)
        Dy = [
            [[self.c5, self.c3, self.c1 / 2, self.Riy / 6], [pyMAOS.unit_manager.ureg.Quantity(0, self.a.units), self.a]],
            [
                [self.c6, self.c4, self.c2 / 2, (self.Riy + self.p) / 6],
                [self.a, self.L],
            ],
        ]
        Dy[0][0] = [i / self.EI for i in Dy[0][0]]
        Dy[1][0] = [i / self.EI for i in Dy[1][0]]
        #print(Dy)
        print("Dy:"); print_quantity_nested_list(Dy, precision=2, width=20, simplify_units=True)

        self.Wx = PiecewisePolynomial()
        # print(self.Wx) # Axial Load Function
        self.Wy = PiecewisePolynomial()  # Vertical Load Function
        # print(self.Wy)
        self.Ax = PiecewisePolynomial()
        # print(self.Ax)
        self.Dx = PiecewisePolynomial()
        #print(self.Dx)
        from pprint import pprint; pprint(Vy); self.Vy = PiecewisePolynomial(Vy)
        print("Vy:\n", self.Vy)


        self.Mz = PiecewisePolynomial(Mz)
        print("Mz:\n", self.Mz)
        self.Sz = PiecewisePolynomial(Sz)
        print("Sz:\n", self.Sz)
        self.Dy = PiecewisePolynomial(Dy)
        print("Dy:\n",self.Dy)

        # Create PiecewisePolynomial2 instances alongside existing ones
        from pyMAOS.loading.PiecewisePolynomial2 import PiecewisePolynomial2

        self.Vy2 = PiecewisePolynomial2(Vy)
        print("Vy2:", self.Vy2, sep="\n")

        self.Mz2 = PiecewisePolynomial2(Mz)
        print("Mz2:", self.Mz2, sep="\n")

        self.Sz2 = PiecewisePolynomial2(Sz)
        print("Sz2:", self.Sz2, sep="\n")

        self.Dy2 = PiecewisePolynomial2(Dy)
        print("Dy2:", self.Dy2, sep="\n")


    def FEF(self):
        p = self.p
        a = self.a
        L = self.L

        # Calculate fixed end moments
        Miz = -1 * (p * a * (a - L) * (a - L)) / (L * L)
        Mjz = -1 * (p * a * a * (a - L)) / (L * L)

        # Calculate fixed end forces
        Riy = self.Riy + (Miz / L) + (Mjz / L)
        Rjy = self.Rjy - (Miz / L) - (Mjz / L)

        # Import dimension constants
        from pyMAOS import INTERNAL_MOMENT_UNIT, INTERNAL_FORCE_UNIT

        # Define expected dimensionalities
        FORCE_DIMENSIONALITY = pyMAOS.unit_manager.ureg.parse_units(pyMAOS.unit_manager.INTERNAL_FORCE_UNIT).dimensionality
        MOMENT_DIMENSIONALITY = pyMAOS.unit_manager.ureg.parse_units(pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT).dimensionality

        # Debug prints showing actual dimensionality
        print(f"DEBUG: Checking dimensions - Miz: {Miz.dimensionality}, Mjz: {Mjz.dimensionality}")
        print(f"DEBUG: Checking dimensions - Riy: {Riy.dimensionality}, Rjy: {Rjy.dimensionality}")

        # Verify moment dimensions
        try:
            Miz.check(MOMENT_DIMENSIONALITY)
            Mjz.check(MOMENT_DIMENSIONALITY)
            print("DEBUG: Moment dimension check passed")
        except pint.DimensionalityError as e:
            print(f"ERROR: Dimension error in moments: {e}")
            # Create correctly dimensioned values as fallback
            if not Miz.check(MOMENT_DIMENSIONALITY):
                print(f"WARNING: Fixing dimensions of Miz from {Miz.dimensionality} to {MOMENT_DIMENSIONALITY}")
                Miz = pyMAOS.unit_manager.ureg.Quantity(Miz.magnitude, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT)
            if not Mjz.check(MOMENT_DIMENSIONALITY):
                print(f"WARNING: Fixing dimensions of Mjz from {Mjz.dimensionality} to {MOMENT_DIMENSIONALITY}")
                Mjz = pyMAOS.unit_manager.ureg.Quantity(Mjz.magnitude, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT)

        # Verify force dimensions
        try:
            Riy.check(FORCE_DIMENSIONALITY)
            Rjy.check(FORCE_DIMENSIONALITY)
            print("DEBUG: Force dimension check passed")
        except pint.DimensionalityError as e:
            print(f"ERROR: Dimension error in forces: {e}")
            # Create correctly dimensioned values as fallback
            if not Riy.check(FORCE_DIMENSIONALITY):
                print(f"WARNING: Fixing dimensions of Riy from {Riy.dimensionality} to {FORCE_DIMENSIONALITY}")
                Riy = pyMAOS.unit_manager.ureg.Quantity(Riy.magnitude, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT)
            if not Rjy.check(FORCE_DIMENSIONALITY):
                print(f"WARNING: Fixing dimensions of Rjy from {Rjy.dimensionality} to {FORCE_DIMENSIONALITY}")
                Rjy = pyMAOS.unit_manager.ureg.Quantity(Rjy.magnitude, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT)

        # Create zeros with appropriate units
        zero_force = pyMAOS.unit_manager.ureg.Quantity(0, Riy.units)

        # Print forces and moments for debugging
        print(f"Point load FEF - Forces: Riy={Riy:.3f}, Rjy={Rjy:.3f}")
        print(f"Point load FEF - Moments: Miz={Miz:.3f}, Mjz={Mjz:.3f}")

        ret_val = [zero_force, Riy, Miz, zero_force, Rjy, Mjz]

        # Final dimension check for return values
        for i, (val, expected_dim) in enumerate(zip(ret_val, [
            FORCE_DIMENSIONALITY, FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY,
            FORCE_DIMENSIONALITY, FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY
        ])):
            try:
                val.check(expected_dim)
            except pint.DimensionalityError as e:
                print(f"ERROR: Dimensionality error in ret_val[{i}]: {e}")
                print(f"  Actual: {val.dimensionality}, Expected: {expected_dim}")

        return ret_val

    def plot_all_ppoly_functions(self, figsize=(10, 12), convert_x_to=None, convert_y_to=None):
        """
        Create a figure with subplots for all PiecewisePolynomial2 functions.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        convert_x_to : pint.Unit, optional
            Convert x values to this unit for plotting
        convert_y_to : dict, optional
            Dictionary mapping function name to unit for conversion, e.g. {'Vy2': 'kN'}

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing all plots
        """
        import matplotlib.pyplot as plt

        # Collect all non-empty PiecewisePolynomial2 objects
        functions = []
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

            # Add vertical lines at key points
            for x in [self.a, self.L]:
                if hasattr(x, 'magnitude'):
                    x_val = x.to(convert_x_to).magnitude if convert_x_to else x.magnitude
                else:
                    x_val = x
                axes[i].axvline(x=x_val, color='gray', linestyle='--', alpha=0.7)

        # Add overall title with sign-preserving formatting for load value
        load_str = f"{self.p:+.3g}"  # Shows + for positive values
        fig.suptitle(f"Beam Analysis (PPoly) for Point Load {load_str} at x={self.a:.3g}", fontsize=16)

        # Adjust spacing
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        # Show the grid on all plots
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.7)

        return fig