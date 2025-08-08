import pint
from typing import TYPE_CHECKING, Any
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial
import numpy as np
import pyMAOS
from pyMAOS import unit_manager

# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from pyMAOS.frame2d import R2Frame


class R2_Axial_Load:
    def __init__(self, p: pint.Quantity, a: pint.Quantity, member: "Any", loadcase="D"):
        self.p = p
        self.a = a
        self.L = member.length

        self.E = member.material.E
        self.A = member.section.Area

        self.EA = self.E * self.A

        self.kind = "AXIAL_POINT"
        self.loadcase = loadcase

        # Simple End Reactions
        self.Rix = -1 * self.p
        self.Rjx = 0

        # Constants of Integration
        p = self.p
        a = self.a

        self.c1 = 0
        self.c2 = p * a

        # Piecewise Functions using wrapped ndarrays
        # Each piecewise function uses the following format:
        # [coefficients_list, domain_bounds]
        # where coefficients_list contains polynomial coefficients in ascending order [c0, c01, c02,...]
        # representing c0 + c01*x + c02*x^2 + ...
        # and domain_bounds are [start_x, end_x] for the applicable region
        # [co....cn x^n] [xa, xb]
        Ax = [
            [[pyMAOS.unit_manager.ureg.Quantity(np.array([-1 * self.Rix.magnitude]), self.Rix.units)],
             [pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT), self.a]],
            [[pyMAOS.unit_manager.ureg.Quantity(np.array([-1 * self.Rix.magnitude - self.p.magnitude]), self.Rix.units)],
             [self.a, self.L]],
        ]

        Dx = [
            [[pyMAOS.unit_manager.ureg.Quantity(np.array([self.c1.magnitude]), self.c1.units),
              pyMAOS.unit_manager.ureg.Quantity(np.array([-1 * self.Rix.magnitude]), self.Rix.units)],
             [pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT), self.a]],
            [[pyMAOS.unit_manager.ureg.Quantity(np.array([self.c2.magnitude]), self.c2.units),
              pyMAOS.unit_manager.ureg.Quantity(np.array([-1 * self.Rix.magnitude - self.p.magnitude]), self.Rix.units)],
             [self.a, self.L]],
        ]

        # Apply EA division using numpy operations
        Dx[0][0] = pyMAOS.unit_manager.ureg.Quantity(
            np.array([coef.magnitude for coef in Dx[0][0]]) / self.EA.magnitude,
            f"{Dx[0][0][0].units}/{self.EA.units}"
        )
        Dx[1][0] = pyMAOS.unit_manager.ureg.Quantity(
            np.array([coef.magnitude for coef in Dx[1][0]]) / self.EA.magnitude,
            f"{Dx[1][0][0].units}/{self.EA.units}"
        )
        print("Dx=",Dx,sep="\n")
        self.Wx = PiecewisePolynomial()  # Axial Load Function
        self.Wy = PiecewisePolynomial()  # Vertical Load Function
        self.Ax = PiecewisePolynomial(Ax); print("Ax:", self.Ax, sep="\n")
        self.Dx = PiecewisePolynomial(Dx); print("Dx:", self.Dx, sep="\n")
        self.Vy = PiecewisePolynomial()
        self.Mz = PiecewisePolynomial()
        self.Sz = PiecewisePolynomial()
        self.Dy = PiecewisePolynomial()

    # def integration_constants(self):
    #     p = self.p
    #     a = self.a
    #
    #     self.c1 = 0
    #     self.c2 = p * a

    def FEF(self):
        p = self.p
        a = self.a
        L = self.L

        Rix = (p * (a - L)) / L
        Rjx = (-1 * p * a) / L

        # Print forces in both SI and display units
        from pyMAOS.pymaos_units import convert_to_display_units
        Rix_display = convert_to_display_units(Rix, 'force')
        Rjx_display = convert_to_display_units(Rjx, 'force')
        import pyMAOS.pymaos_units as units
        # Then use units.DISPLAY_UNITS which will reflect the current value
        print(f"Axial reactions - SI: Rix={Rix:.3f} N, Rjx={Rjx:.3f} N")
        print(f"Axial reactions - Display: Rix={Rix_display:.3f} {units.DISPLAY_UNITS['force']}, "
              f"Rjx={Rjx_display:.3f} {units.DISPLAY_UNITS['force']}")

        # Return a properly wrapped ndarray with units
        return pyMAOS.unit_manager.ureg.Quantity(
            np.array([Rix.magnitude, 0, 0, Rjx.magnitude, 0, 0]),
            pyMAOS.unit_manager.INTERNAL_FORCE_UNIT
        )

    def __str__(self):
        """
        String representation of an axial point load.
        """
        return (f"Axial Point Load ({self.loadcase}): "
                f"p={self.p:.3f} at x={self.a:.3f} "
                f"(on member of length {self.L:.3f})")
    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Prints detailed analysis of axial response with ASCII charts.

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
        import sys
        from pyMAOS import unit_manager

        # Get current unit system directly from the manager
        current_units = unit_manager.get_current_units()
        system_name = unit_manager.get_system_name()

        print(f"\n===== DETAILED ANALYSIS FOR {self.__str__()} =====")
        print(f"Axial reactions: Rix = {self.Rix:.3f}, Rjx = {self.Rjx:.3f}")

        # Define regions for before and after the load point
        regions = [(unit_manager.ureg.Quantity(0, unit_manager.INTERNAL_LENGTH_UNIT), self.a),
                   (self.a, self.L)]
        region_names = ["Before Load [0 to a]", "After Load [a to L]"]

        # Create sampling points
        all_x = []
        for i, (start, end) in enumerate(regions):
            if end > start:  # Only if region has non-zero width
                points = [start + j*(end-start)/num_points for j in range(num_points+1)]
                print(f"Region {i+1} ({region_names[i]}): {points}")
                # Don't duplicate boundary points
                if i > 0 and len(all_x) > 0:
                    points = points[1:]
                all_x.extend(points)

        # Convert to numpy array
        x_array = np.array(all_x, dtype=object)

        # Calculate function values using vectorized evaluation if available
        ax_values = self.Ax.evaluate_vectorized(x_array) if hasattr(self.Ax, 'evaluate_vectorized') else [self.Ax.evaluate(x) for x in all_x]
        dx_values = self.Dx.evaluate_vectorized(x_array) if hasattr(self.Dx, 'evaluate_vectorized') else [self.Dx.evaluate(x) for x in all_x]

        # Print ASCII charts
        self._print_ascii_chart("Axial Force (Ax)", all_x, ax_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Axial Displacement (Dx)", all_x, dx_values, regions, chart_width, chart_height)

        # Print table of values at key points
        print("\n===== VALUES AT KEY POINTS =====")
        print(f"{'Position':15} {'Axial Force':20} {'Axial Displacement':20}")
        print("-" * 60)
        for x in [0, self.a, self.L]:
            print(f"{x:15.3f} {self.Ax.evaluate(x):20.3f} {self.Dx.evaluate(x):20.3e}")

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
        from pyMAOS import unit_manager
        print(f"Region boundaries: [{unit_manager.ureg.Quantity(0, self.a.units)}, {self.a:.2f}, {self.L:.2f}]")

class R2_Axial_Linear_Load:
    def __init__(self, w1, w2, a, b, member, loadcase="D"):
        self.w1 = w1
        self.w2 = w2
        self.a = a
        self.b = b
        self.c = b - a
        self.L = member.length

        self.E = member.material.E
        self.A = member.section.Area

        self.EA = self.E * self.A

        self.kind = "AXIAL_LINE"
        self.loadcase = loadcase

        # Simple End Reactions
        self.W = 0.5 * self.c * (self.w2 + self.w1)

        self.Rix = -1 * self.W
        self.Rjx = 0

        # Constants of Integration
        self.integration_constants()

        # Piecewise Functions
        # [co....cn x^n] [xa, xb]
        Wx = [
            [[0], [0, self.a]],
            [
                [
                    ((-1 * self.a * self.w2) - (self.c * self.w1) - (self.a * self.w1))
                    / self.c,
                    (self.w2 - self.w1) / self.c,
                ],
                [self.a, self.b],
            ],
            [[0], [self.b, self.L]],
        ]

        Ax = [
            [[-1 * self.Rix], [0, self.a]],
            [
                [
                    self.c1,
                    (self.a * self.w2 - self.b * self.w1) / (self.c),
                    -1 * (self.w2 - self.w1) / (2 * self.c),
                ],
                [self.a, self.b],
            ],
            [[-1 * self.Rix - self.W], [self.b, self.L]],
        ]

        Dx = [
            [[self.c2, -1 * self.Rix], [0, self.a]],
            [
                [
                    self.c3,
                    self.c1,
                    ((self.a * self.w2 - self.b * self.w1)) / (2 * self.c),
                    -1 * ((self.w2 - self.w1)) / (6 * self.c),
                ],
                [self.a, self.b],
            ],
            [[self.c4, -1 * self.Rix - self.W], [self.b, self.L]],
        ]

        Dx[0][0] = [i / self.EA for i in Dx[0][0]]
        Dx[1][0] = [i / self.EA for i in Dx[1][0]]
        Dx[2][0] = [i / self.EA for i in Dx[2][0]]

        self.Wx = PiecewisePolynomial(Wx)  # Axial Load Function
        self.Wy = PiecewisePolynomial()  # Vertical Load Function
        self.Ax = PiecewisePolynomial(Ax)
        self.Dx = PiecewisePolynomial(Dx)
        self.Vy = PiecewisePolynomial()
        self.Mz = PiecewisePolynomial()
        self.Sz = PiecewisePolynomial()
        self.Dy = PiecewisePolynomial()

    def integration_constants(self):
        w1 = self.w1
        w2 = self.w2
        a = self.a
        b = self.b
        Ri = self.Rix

        self.c1 = -(
            (a * a * w2 - 2 * a * b * w1 + a * a * w1 + 2 * Ri * b - 2 * Ri * a)
            / (2 * (b - a))
        )
        self.c2 = 0
        self.c3 = (a * a * (a * w2 - 3 * b * w1 + 2 * a * w1)) / (6 * (b - a))
        self.c4 = (
            (2 * b * b - a * b - a * a) * w2 + (b * b + a * b - 2 * a * a) * w1
        ) / 6

    def FEF(self):
        w1 = self.w1
        w2 = self.w2
        a = self.a
        b = self.b
        L = self.L

        Rix = (
            (b - a)
            * (2 * b * w2 + a * w2 - 3 * L * w2 + b * w1 + 2 * a * w1 - 3 * L * w1)
        ) / (6 * L)
        Rjx = -1 * (((b - a) * (2 * b * w2 + a * w2 + b * w1 + 2 * a * w1)) / (6 * L))

        # Return a properly wrapped ndarray with units
        return pyMAOS.unit_manager.ureg.Quantity(
            np.array([Rix.magnitude, 0, 0, Rjx.magnitude, 0, 0]),
            pyMAOS.unit_manager.INTERNAL_FORCE_UNIT
        )

    def __str__(self):
        """
        String representation of an axial linear load.
        """
        return (f"Axial Linear Load ({self.loadcase}): "
                f"w1={self.w1:.3f}, w2={self.w2:.3f}, "
                f"from x={self.a:.3f} to x={self.b:.3f} "
                f"(on member of length {self.L:.3f})")

    def print_detailed_analysis(self, num_points=10, chart_width=60, chart_height=15):
        """
        Prints detailed analysis of axial response with ASCII charts.

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
        import sys
        from pyMAOS import unit_manager

        # Get current unit system directly from the manager
        current_units = unit_manager.get_current_units()
        system_name = unit_manager.get_system_name()

        print(f"\n===== DETAILED ANALYSIS FOR {self.__str__()} =====")
        print(f"Axial reactions: Rix = {self.Rix:.3f}, Rjx = {self.Rjx:.3f}")

        # Define regions for before and after the load point
        regions = [(unit_manager.ureg.Quantity(0, unit_manager.INTERNAL_LENGTH_UNIT), self.a),
                   (self.a, self.L)]
        region_names = ["Before Load [0 to a]", "After Load [a to L]"]

        # Create sampling points
        all_x = []
        for i, (start, end) in enumerate(regions):
            if end > start:  # Only if region has non-zero width
                points = [start + j*(end-start)/num_points for j in range(num_points+1)]
                print(f"Region {i+1} ({region_names[i]}): {points}")
                # Don't duplicate boundary points
                if i > 0 and len(all_x) > 0:
                    points = points[1:]
                all_x.extend(points)

        # Convert to numpy array
        x_array = np.array(all_x, dtype=object)

        # Calculate function values using vectorized evaluation if available
        ax_values = self.Ax.evaluate_vectorized(x_array) if hasattr(self.Ax, 'evaluate_vectorized') else [self.Ax.evaluate(x) for x in all_x]
        dx_values = self.Dx.evaluate_vectorized(x_array) if hasattr(self.Dx, 'evaluate_vectorized') else [self.Dx.evaluate(x) for x in all_x]

        # Print ASCII charts
        self._print_ascii_chart("Axial Force (Ax)", all_x, ax_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Axial Displacement (Dx)", all_x, dx_values, regions, chart_width, chart_height)

        # Print table of values at key points
        print("\n===== VALUES AT KEY POINTS =====")
        print(f"{'Position':15} {'Axial Force':20} {'Axial Displacement':20}")
        print("-" * 60)
        for x in [0, self.a, self.L]:
            print(f"{x:15.3f} {self.Ax.evaluate(x):20.3f} {self.Dx.evaluate(x):20.3e}")

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
        from pyMAOS import unit_manager
        print(f"Region boundaries: [{unit_manager.ureg.Quantity(0, self.a.units)}, {self.a:.2f}, {self.L:.2f}]")