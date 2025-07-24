import pint

from pyMAOS.display_utils import display_node_load_vector_in_units
from pyMAOS.frame2d import R2Frame
from pyMAOS.loading.polynomial import Piecewise_Polynomial

class R2_Linear_Load:
    def __init__(self, w1: pint.Quantity, w2: pint.Quantity, a: pint.Quantity, b: pint.Quantity, member: R2Frame, loadcase="D"):
        self.w1 = w1
        self.w2 = w2
        self.a = a
        self.b = b
        self.c = b - a
        self.L = member.length

        self.E = member.material.E
        self.I = member.section.Ixx

        self.EI = self.E * self.I

        self.kind = "LINE"
        self.loadcase = loadcase

        # Constants of Integration
        self.integration_constants()

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

        Vy = [
            [[self.c1], [0, self.a]],
            [
                [
                    self.c2,
                    self.w1
                    + ((self.a * self.w1) / self.c)
                    - ((self.a * self.w2) / self.c),
                    (self.w2 / (2 * self.c)) - (self.w1 / (2 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c3], [self.b, self.L]],
        ]

        Mz = [
            [[self.c4, self.c1], [0, self.a]],
            [
                [
                    self.c5,
                    self.c2,
                    (self.w1 / 2)
                    + ((self.a * self.w1) / (2 * self.c))
                    - ((self.a * self.w2) / (2 * self.c)),
                    (self.w2 / (6 * self.c)) - (self.w1 / (6 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c6, self.c3], [self.b, self.L]],
        ]

        Sz = [
            [[self.c7, self.c4, 0.5 * self.c1], [0, self.a]],
            [
                [
                    self.c8,
                    self.c5,
                    0.5 * self.c2,
                    (self.w1 / 6)
                    + ((self.a * self.w1) / (6 * self.c))
                    - ((self.a * self.w2) / (6 * self.c)),
                    (self.w2 / (24 * self.c)) - (self.w1 / (24 * self.c)),
                ],
                [self.a, self.b],
            ],
            [[self.c9, self.c6, 0.5 * self.c3], [self.b, self.L]],
        ]

        Dy = [
            [[self.c10, self.c7, 0.5 * self.c4, self.c1 / 6], [0, self.a]],
            [
                [
                    self.c11,
                    self.c8,
                    0.5 * self.c5,
                    self.c2 / 6,
                    (self.w1 / 24)
                    + ((self.a * self.w1) / (24 * self.c))
                    - ((self.a * self.w2) / (24 * self.c)),
                    (self.w2 / (120 * self.c)) - (self.w1 / (120 * self.c)),
                ],
                [self.a, self.b],
            ],
            [
                [self.c12, self.c9, 0.5 * self.c6, self.c3 / 6],
                [self.b, self.L],
            ],
        ]

        Sz[0][0] = [i / self.EI for i in Sz[0][0]]
        Sz[1][0] = [i / self.EI for i in Sz[1][0]]
        Sz[2][0] = [i / self.EI for i in Sz[2][0]]

        Dy[0][0] = [i / self.EI for i in Dy[0][0]]
        Dy[1][0] = [i / self.EI for i in Dy[1][0]]
        Dy[2][0] = [i / self.EI for i in Dy[2][0]]

        self.Wx = Piecewise_Polynomial()  # Axial Load Function
        self.Wy = Piecewise_Polynomial(Wy)  # Vertical Load Function
        self.Ax = Piecewise_Polynomial()
        self.Dx = Piecewise_Polynomial()
        self.Vy = Piecewise_Polynomial(Vy)
        self.Mz = Piecewise_Polynomial(Mz)
        self.Sz = Piecewise_Polynomial(Sz)
        self.Dy = Piecewise_Polynomial(Dy)

    def integration_constants(self):
        """
        Calculate the integration constants for beam deflection equations.
        
        These constants (c1-c12) are determined by enforcing boundary conditions:
        - Continuity of shear, moment, slope, and deflection at load discontinuities
        - Zero displacement and rotation at member ends (for fixed-end conditions)
        - Equilibrium of forces and moments
        
        The constants are used in the piecewise polynomial functions that define:
        - Vy: Shear force distribution
        - Mz: Bending moment distribution
        - Sz: Slope (rotation) distribution
        - Dy: Deflection distribution
        """
        w1 = self.w1
        w2 = self.w2
        a = self.a
        b = self.b
        L = self.L

        # Constants for the shear force function (Vy)
        self.c1 = (
            (((2 * b * b) + ((-a - 3 * L) * b) - (a * a) + (3 * L * a)) * w2)
            + (((b * b) + ((a - 3 * L) * b) - (2 * a * a) + (3 * L * a)) * w1)
        ) / (6 * L)
        self.c2 = (
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
        self.c3 = (
            ((2 * b * b - a * b - a * a) * w2) + ((b * b + a * b - 2 * a * a) * w1)
        ) / (6 * L)
        
        # Constants for the bending moment function (Mz)
        self.c4 = 0  # Zero moment at x=0 for fixed-end condition
        self.c5 = (
            -1
            * ((a * a * a * w2) + ((2 * a * a * a - 3 * a * a * b) * w1))
            / (6 * b - 6 * a)
        )
        self.c6 = (
            -1
            * ((2 * b * b - a * b - a * a) * w2 + (b * b + a * b - 2 * a * a) * w1)
            / 6
        )
        
        # Constants for the slope function (Sz)
        self.c7 = (
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
        self.c8 = (
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
        self.c9 = (
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
        self.c10 = 0  # Zero deflection at x=0 for fixed-end condition
        self.c11 = (
            -1
            * (
                a * a * a * a * a * w2
                + (4 * a * a * a * a * a - 5 * a * a * a * a * b) * w1
            )
            / (120 * b - 120 * a)
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

    def FEF(self):
        L = self.L

        c3 = self.c3
        c6 = self.c6
        c7 = self.c7
        c9 = self.c9

        Miz = -1 * (c3 * L * L + 2 * c6 * L + 2 * c9 + 4 * c7) / L
        Mjz = -1 * (2 * c3 * L * L + 4 * c6 * L + 4 * c9 + 2 * c7) / L
        Riy = self.Riy + (Miz / L) + (Mjz / L)
        Rjy = self.Rjy - (Miz / L) - (Mjz / L)
        
        # Print forces and moments in both SI and display units
        from pyMAOS.units_mod import convert_to_display_units
        from pyMAOS.units_mod import unit_manager
        # Get current unit system directly from the manager
        current_units = unit_manager.get_current_units()
        system_name = unit_manager.get_system_name()
        Riy_display = convert_to_display_units(Riy, 'force')
        Rjy_display = convert_to_display_units(Rjy, 'force')
        Miz_display = convert_to_display_units(Miz, 'moment')
        Mjz_display = convert_to_display_units(Mjz, 'moment')
        
        print(f"Vertical reactions - SI: Riy={Riy:.3f} N, Rjy={Rjy:.3f} N")
        print(f"Vertical reactions - Display: Riy={Riy_display:.3f} {current_units['force']}, "
              f"Rjy={Rjy_display:.3f} {current_units['force']}")
        print(f"Moments - SI: Miz={Miz:.3f} N*m, Mjz={Mjz:.3f} N*m")
        print(f"Moments - Display: Miz={Miz_display:.3f} {current_units['moment']}, "
              f"Mjz={Mjz_display:.3f} {current_units['moment']}")
        
        ret_val = [0, Riy, Miz, 0, Rjy, Mjz]
        print(f"FEF distributed load results for Load Case {self.loadcase}:\n", ret_val)
        display_node_load_vector_in_units(ret_val[0:3], "node_i",
                                          force_unit='klbf', 
                                          length_unit='in')
        display_node_load_vector_in_units(ret_val[3:6], "node_j",
                                          force_unit='klbf',
                                          length_unit='in')

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
        from pyMAOS.units_mod import convert_to_display_units
        from pyMAOS.units_mod import unit_manager
        # Get current unit system directly from the manager
        current_units = unit_manager.get_current_units()
        system_name = unit_manager.get_system_name()
        print(f"\n===== DETAILED ANALYSIS FOR {self.__str__()} =====")
        print(f"Total Load W = {self.W:.3f} N ({convert_to_display_units(self.W, 'force'):.3f} {current_units['force']})")
        print(f"Load centroid from left: {self.a + self.cbar:.3f} m")
        print(f"Reactions: Riy = {self.Riy:.3f} N, Rjy = {self.Rjy:.3f} N")
        print(f"          ({convert_to_display_units(self.Riy, 'force'):.3f} {current_units['force']}, "
              f"{convert_to_display_units(self.Rjy, 'force'):.3f} {current_units['force']})")
        
        # Sample points across all regions
        regions = [(0, self.a), (self.a, self.b), (self.b, self.L)]
        region_names = ["Before Load [0 to a]", "Loaded Region [a to b]", "After Load [b to L]"]
        
        # Create sampling points
        all_x = []
        for i, (start, end) in enumerate(regions):
            if end > start:  # Only if region has non-zero width
                points = [start + j*(end-start)/num_points for j in range(num_points+1)]
                # Don't duplicate boundary points
                if i > 0 and len(all_x) > 0:
                    points = points[1:]
                all_x.extend(points)
        
        # Calculate function values
        wy_values = [self.Wy.evaluate(x) for x in all_x]
        vy_values = [self.Vy.evaluate(x) for x in all_x]
        mz_values = [self.Mz.evaluate(x) for x in all_x]
        sz_values = [self.Sz.evaluate(x) for x in all_x]
        dy_values = [self.Dy.evaluate(x) for x in all_x]
        
        # Print ASCII charts
        self._print_ascii_chart("Distributed Load (Wy)", all_x, wy_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Shear Force (Vy)", all_x, vy_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Bending Moment (Mz)", all_x, mz_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Rotation (Sz)", all_x, sz_values, regions, chart_width, chart_height)
        self._print_ascii_chart("Deflection (Dy)", all_x, dy_values, regions, chart_width, chart_height)
        
        # Print table of values at region boundaries
        print("\n===== VALUES AT KEY POINTS =====")
        print(f"{'Position (m)':<15} {'Load (N/m)':<15} {'Shear (N)':<15} {'Moment (N*m)':<15} {'Rotation (rad)':<15} {'Deflection (m)':<15}")
        print("-" * 90)
        for x in [0, self.a, self.b, self.L]:
            print(f"{x:<15.3f} {self.Wy.evaluate(x):<15.3f} {self.Vy.evaluate(x):<15.3f} {self.Mz.evaluate(x):<15.3f} "
                  f"{self.Sz.evaluate(x):<15.3e} {self.Dy.evaluate(x):<15.3e}")
    
    def _print_ascii_chart(self, title, x_values, y_values, regions, width=60, height=15):
        """Helper method to print an ASCII chart of the data"""
        if not y_values:
            return
            
        print(f"\n--- {title} ---")
        
        # Find min and max values
        min_y = min(y_values)
        max_y = max(y_values)
        if min_y == max_y:  # Avoid division by zero
            min_y -= 1
            max_y += 1
            
        # Create the chart grid
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw x-axis
        axis_pos = height - int(height * (-min_y) / (max_y - min_y)) if min_y < 0 and max_y > 0 else height - 1
        axis_pos = max(0, min(height - 1, axis_pos))
        for i in range(width):
            chart[axis_pos][i] = '-'
            
        # Draw y-axis
        for i in range(height):
            chart[i][0] = '|'
            
        # Mark region boundaries
        region_positions = [0]
        for start, end in regions:
            if end not in region_positions:
                region_positions.append(end)
        
        region_x_positions = []
        for pos in region_positions:
            if pos == 0:
                x_pos = 0
            else:
                x_pos = int(width * pos / self.L)
                x_pos = min(x_pos, width - 1)
            region_x_positions.append(x_pos)
            
        for x_pos in region_x_positions:
            for i in range(height):
                if i != axis_pos:
                    chart[i][x_pos] = '|'
                else:
                    chart[i][x_pos] = '+'
        
        # Plot the data
        for i in range(len(x_values)):
            x = x_values[i]
            y = y_values[i]
            
            # Convert to chart coordinates
            chart_x = int(width * x / self.L)
            chart_y = height - 1 - int((height - 1) * (y - min_y) / (max_y - min_y))
            
            chart_x = max(0, min(width - 1, chart_x))
            chart_y = max(0, min(height - 1, chart_y))
            
            # Draw data point
            chart[chart_y][chart_x] = '*'
            
            # Connect with previous point
            if i > 0:
                prev_x = int(width * x_values[i-1] / self.L)
                prev_y = height - 1 - int((height - 1) * (y_values[i-1] - min_y) / (max_y - min_y))
                
                prev_x = max(0, min(width - 1, prev_x))
                prev_y = max(0, min(height - 1, prev_y))
                
                # Simple line drawing
                if prev_x != chart_x or prev_y != chart_y:
                    # Draw a connecting line (very simple algorithm)
                    steps = max(abs(chart_x - prev_x), abs(chart_y - prev_y))
                    if steps > 0:
                        for j in range(1, steps):
                            connect_x = prev_x + int(j * (chart_x - prev_x) / steps)
                            connect_y = prev_y + int(j * (chart_y - prev_y) / steps)
                            if 0 <= connect_x < width and 0 <= connect_y < height and chart[connect_y][connect_x] == ' ':
                                chart[connect_y][connect_x] = '.'
        
        # Draw chart
        print("+" + "-" * (width + 2) + "+")
        print(f"| {title.center(width)} |")
        print("+" + "-" * (width + 2) + "+")
        for row in chart:
            print("|", ''.join(row), "|")
        print("+" + "-" * (width + 2) + "+")
        
        # Print legend
        print(f"Min value: {min_y:.3e}, Max value: {max_y:.3e}")
        print(f"x-axis: 0 to {self.L:.3f} m")
        print(f"Region boundaries: [0, {self.a:.3f}, {self.b:.3f}, {self.L:.3f}]")