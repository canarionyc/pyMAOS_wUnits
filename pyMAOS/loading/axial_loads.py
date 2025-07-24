import pint

from pyMAOS.frame2d import R2Frame
from pyMAOS.loading.polynomial import Piecewise_Polynomial


class R2_Axial_Load:
    def __init__(self, p: pint.Quantity, a: pint.Quantity, member: R2Frame, loadcase="D"):
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
        self.integration_constants()

        # Piecewise Functions
        # Each piecewise function uses the following format:
        # [coefficients_list, domain_bounds]
        # where coefficients_list contains polynomial coefficients in ascending order [c0, c1, c2,...] 
        # representing c0 + c1*x + c2*x^2 + ...
        # and domain_bounds are [start_x, end_x] for the applicable region
        # [co....cn x^n] [xa, xb]
        Ax = [
            [[-1 * self.Rix], [0, self.a]],
            [[-1 * self.Rix - self.p], [self.a, self.L]],
        ]

        Dx = [
            [[self.c1, -1 * self.Rix], [0, self.a]],
            [[self.c2, -1 * self.Rix - self.p], [self.a, self.L]],
        ]

        Dx[0][0] = [i / self.EA for i in Dx[0][0]]
        Dx[1][0] = [i / self.EA for i in Dx[1][0]]

        self.Wx = Piecewise_Polynomial()  # Axial Load Function
        self.Wy = Piecewise_Polynomial()  # Vertical Load Function
        self.Ax = Piecewise_Polynomial(Ax)
        self.Dx = Piecewise_Polynomial(Dx)
        self.Vy = Piecewise_Polynomial()
        self.Mz = Piecewise_Polynomial()
        self.Sz = Piecewise_Polynomial()
        self.Dy = Piecewise_Polynomial()

    def integration_constants(self):
        p = self.p
        a = self.a

        self.c1 = 0
        self.c2 = p * a

    def FEF(self):
        p = self.p
        a = self.a
        L = self.L

        Rix = (p * (a - L)) / L
        Rjx = (-1 * p * a) / L

        # Print forces in both SI and display units
        from pyMAOS.units_mod import convert_to_display_units
        Rix_display = convert_to_display_units(Rix, 'force')
        Rjx_display = convert_to_display_units(Rjx, 'force')
        import pyMAOS.units_mod as units
        # Then use units.DISPLAY_UNITS which will reflect the current value
        print(f"Axial reactions - SI: Rix={Rix:.3f} N, Rjx={Rjx:.3f} N")
        print(f"Axial reactions - Display: Rix={Rix_display:.3f} {units.DISPLAY_UNITS['force']}, "
              f"Rjx={Rjx_display:.3f} {units.DISPLAY_UNITS['force']}")

        return [Rix, 0, 0, Rjx, 0, 0]

    def __str__(self):
        """
        String representation of an axial point load.
        """
        return (f"Axial Point Load ({self.loadcase}): "
                f"p={self.p:.3f} at x={self.a:.3f} "
                f"(on member of length {self.L:.3f})")


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

        self.Wx = Piecewise_Polynomial(Wx)  # Axial Load Function
        self.Wy = Piecewise_Polynomial()  # Vertical Load Function
        self.Ax = Piecewise_Polynomial(Ax)
        self.Dx = Piecewise_Polynomial(Dx)
        self.Vy = Piecewise_Polynomial()
        self.Mz = Piecewise_Polynomial()
        self.Sz = Piecewise_Polynomial()
        self.Dy = Piecewise_Polynomial()

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

        return [Rix, 0, 0, Rjx, 0, 0]

    def __str__(self):
        """
        String representation of an axial linear load.
        """
        return (f"Axial Linear Load ({self.loadcase}): "
                f"w1={self.w1:.3f}, w2={self.w2:.3f}, "
                f"from x={self.a:.3f} to x={self.b:.3f} "
                f"(on member of length {self.L:.3f})")