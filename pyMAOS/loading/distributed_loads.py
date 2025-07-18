from pyMAOS.display_utils import display_node_load_vector_in_units
from pyMAOS.loading.polynomial import Piecewise_Polynomial

class R2_Linear_Load:
    def __init__(self, w1, w2, a, b, member, loadcase="D"):
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
        from pyMAOS.units import convert_to_display_units, DISPLAY_UNITS
        Riy_display = convert_to_display_units(Riy, 'force')
        Rjy_display = convert_to_display_units(Rjy, 'force')
        Miz_display = convert_to_display_units(Miz, 'moment')
        Mjz_display = convert_to_display_units(Mjz, 'moment')
        
        print(f"Vertical reactions - SI: Riy={Riy:.3f} N, Rjy={Rjy:.3f} N")
        print(f"Vertical reactions - Display: Riy={Riy_display:.3f} {DISPLAY_UNITS['force']}, "
              f"Rjy={Rjy_display:.3f} {DISPLAY_UNITS['force']}")
        print(f"Moments - SI: Miz={Miz:.3f} N*m, Mjz={Mjz:.3f} N*m")
        print(f"Moments - Display: Miz={Miz_display:.3f} {DISPLAY_UNITS['moment']}, "
              f"Mjz={Mjz_display:.3f} {DISPLAY_UNITS['moment']}")
        
        ret_val = [0, Riy, Miz, 0, Rjy, Mjz]
        print(f"FEF distributed load results for Load Case {self.loadcase}:\n", ret_val)
        display_node_load_vector_in_units(ret_val[0:3], 
                                          force_unit='klbf', 
                                          length_unit='in')
        display_node_load_vector_in_units(ret_val[3:6], 
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