import pint
from typing import TYPE_CHECKING, Any
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial
from pprint import pprint
from display_utils import print_quantity_nested_list
import pyMAOS
# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from pyMAOS.frame2d import R2Frame

class R2_Load_Base:
    """Base class for point loads and moments in R2 frames."""
    
    def __init__(self, a, member, loadcase="D"):
        """Initialize common attributes for all loads."""
        self.a = a
        self.L = member.length
        
        self.E = member.material.E
        self.I = member.section.Ixx
        self.EI = self.E * self.I
        
        self.kind = "LOAD_BASE"  # Will be overridden by child classes
        self.loadcase = loadcase
        
        # Initialize piecewise polynomials to None - will be set by child classes
        self.Wx = PiecewisePolynomial()  # Axial Load Function
        self.Wy = PiecewisePolynomial()  # Vertical Load Function
        self.Ax = PiecewisePolynomial()  # Axial Force Function
        self.Dx = PiecewisePolynomial()  # Axial Displacement Function
        self.Vy = None  # Shear Function
        self.Mz = None  # Moment Function
        self.Sz = None  # Rotation Function
        self.Dy = None  # Deflection Function
    
    def _print_ascii_chart(self, title, x_values, y_values, regions, width=60, height=15):
        """
        Helper method to print an ASCII chart of data with proper unit handling.
        """
        import numpy as np

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
        print(f"Region boundaries: [0, {self.a:.2f}, {self.L:.2f}]")
    
    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Prints detailed analysis of the load response with ASCII charts.
        
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
        from pyMAOS import unit_manager
    
        print(f"\n===== DETAILED ANALYSIS FOR {self.__str__()} =====")
        
        # Define regions for before and after the load point
        regions = [(0, self.a), (self.a, self.L)]
        region_names = ["Before Load [0 to a]", "After Load [a to L]"]
    
        # Create sampling points for each region
        all_x = []
        for i, (start, end) in enumerate(regions):
            if end > start:  # Only if region has non-zero width
                points = [start + j*(end-start)/num_points for j in range(num_points+1)]
                # Don't duplicate boundary points
                if i > 0 and len(all_x) > 0:
                    points = points[1:]
                all_x.extend(points)
    
        # Calculate function values for existing functions
        if self.Vy is not None:
            vy_values = [self.Vy.evaluate(x) for x in all_x]
            self._print_ascii_chart("Shear Force (Vy)", all_x, vy_values, regions, chart_width, chart_height)
        
        if self.Mz is not None:
            mz_values = [self.Mz.evaluate(x) for x in all_x]
            self._print_ascii_chart("Bending Moment (Mz)", all_x, mz_values, regions, chart_width, chart_height)
        
        if self.Sz is not None:
            sz_values = [self.Sz.evaluate(x) for x in all_x]
            self._print_ascii_chart("Rotation (Sz)", all_x, sz_values, regions, chart_width, chart_height)
        
        if self.Dy is not None:
            dy_values = [self.Dy.evaluate(x) for x in all_x]
            self._print_ascii_chart("Deflection (Dy)", all_x, dy_values, regions, chart_width, chart_height)
    
        # Print table of values at key points if all functions exist
        if all(f is not None for f in [self.Vy, self.Mz, self.Sz, self.Dy]):
            print("\n===== VALUES AT KEY POINTS =====")
            print(f"{'Position':15} {'Shear':15} {'Moment':15} {'Rotation':15} {'Deflection':15}")
            print("-" * 75)
            for x in [0, self.a, self.L]:
                print(f"{x:15.3f} {self.Vy.evaluate(x):15.3f} {self.Mz.evaluate(x):15.3f} {self.Sz.evaluate(x):15.3e} {self.Dy.evaluate(x):15.3e}")
    
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
            Dictionary mapping function name to unit for conversion, e.g. {'Vy': 'kN'}

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing all plots
        """
        import matplotlib.pyplot as plt

        # Collect all non-empty PiecewisePolynomial2 objects
        functions = []
        if hasattr(self, 'Vy') and self.Vy2.ppoly is not None:
            functions.append(('Vy', self.Vy2, 'red', 'Shear Force'))
        if hasattr(self, 'Mz') and self.Mz2.ppoly is not None:
            functions.append(('Mz', self.Mz2, 'green', 'Bending Moment'))
        if hasattr(self, 'Sz') and self.Sz2.ppoly is not None:
            functions.append(('Sz', self.Sz2, 'purple', 'Rotation'))
        if hasattr(self, 'Dy') and self.Dy2.ppoly is not None:
            functions.append(('Dy', self.Dy2, 'orange', 'Deflection'))

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

        # Add overall title
        fig.suptitle(f"Beam Analysis (PPoly) for {self.__str__()}", fontsize=16)

        # Adjust spacing
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)

        # Show the grid on all plots
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.7)

        return fig
    
    def FEF(self):
        """
        Compute and return the fixed end forces.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement FEF method")
    
    def __str__(self):
        """String representation should be implemented by each child class."""
        raise NotImplementedError("Subclasses must implement __str__")


class R2_Point_Moment(R2_Load_Base):
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
        # Call parent initializer
        super().__init__(a, member, loadcase)
        
        self.M = M
        self.kind = "MOMENT"

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
        
        # Initialize piecewise polynomials
        self.Vy = PiecewisePolynomial(Vy)
        self.Mz = PiecewisePolynomial(Mz)
        self.Sz = PiecewisePolynomial(Sz)
        self.Dy = PiecewisePolynomial(Dy)

        # Create PiecewisePolynomial2 instances if the class is available
        try:
            from pyMAOS.loading.PiecewisePolynomial2 import PiecewisePolynomial2
            self.Vy2 = PiecewisePolynomial2(Vy)
            self.Mz2 = PiecewisePolynomial2(Mz)
            self.Sz2 = PiecewisePolynomial2(Sz)
            self.Dy2 = PiecewisePolynomial2(Dy)
        except ImportError:
            pass

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
        zero_force = 0 * pyMAOS.unit_manager.ureg(Riy.units)

        return [zero_force, Riy, Miz, zero_force, Rjy, Mjz]

    def __str__(self):
        """
        String representation of a point moment.
        """
        return (f"Point Moment ({self.loadcase}): "
                f"M={self.M:.3f} at x={self.a:.3f} "
                f"(on member of length {self.L:.3f})")

    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Extended version of the base class method with moment-specific information.
        """
        print(f"Point moment of {self.M:.3f} at x={self.a:.3f}")
        print(f"Member length: {self.L:.3f}")
        print(f"Vertical reactions: Riy = {self.Riy:.3f}, Rjy = {self.Rjy:.3f}")
        
        # Call the parent class method to do the actual analysis
        super().print_detailed_analysis(num_points, chart_width, chart_height)


class R2_Point_Load(R2_Load_Base):
    def __init__(self, p: pint.Quantity, a: pint.Quantity, member: "Any", loadcase="D"):
        # Call parent initializer
        super().__init__(a, member, loadcase)
        
        self.p = p
        self.kind = "POINT"

        # Calculate constants of integration
        L = self.L
        
        from pyMAOS import INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT, INTERNAL_MOMENT_UNIT
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
        print("Dy:"); print_quantity_nested_list(Dy, precision=2, width=20, simplify_units=True)

        # Initialize piecewise polynomials
        self.Vy = PiecewisePolynomial(Vy)
        print("Vy:\n", self.Vy)
        
        self.Mz = PiecewisePolynomial(Mz)
        print("Mz:\n", self.Mz)
        
        self.Sz = PiecewisePolynomial(Sz)
        print("Sz:\n", self.Sz)
        
        self.Dy = PiecewisePolynomial(Dy)
        print("Dy:\n", self.Dy)

        # Create PiecewisePolynomial2 instances if the class is available
        try:
            from pyMAOS.loading.PiecewisePolynomial2 import PiecewisePolynomial2

            self.Vy2 = PiecewisePolynomial2(Vy)
            print("Vy:", self.Vy2, sep="\n")

            self.Mz2 = PiecewisePolynomial2(Mz)
            print("Mz:", self.Mz2, sep="\n")

            self.Sz2 = PiecewisePolynomial2(Sz)
            print("Sz:", self.Sz2, sep="\n")

            self.Dy2 = PiecewisePolynomial2(Dy)
            print("Dy:", self.Dy2, sep="\n")
        except ImportError:
            pass

    def __str__(self):
        """
        String representation of a point load.
        """
        return (f"Point Load ({self.loadcase}): "
                f"p={self.p:.3f} at x={self.a:.3f} "
                f"(on member of length {self.L:.3f})")

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

    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Extended version of the base class method with point load specific information.
        """
        print(f"Point load of {self.p:.3f} at x={self.a:.3f}")
        print(f"Member length: {self.L:.3f}")
        print(f"Vertical reactions: Riy = {self.Riy:.3f}, Rjy = {self.Rjy:.3f}")
        
        # Call the parent class method to do the actual analysis
        super().print_detailed_analysis(num_points, chart_width, chart_height)
