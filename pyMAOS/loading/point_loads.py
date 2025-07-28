import pint
from typing import TYPE_CHECKING, Any
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial
from pprint import pprint
from display_utils import print_quantity_nested_list
# Use TYPE_CHECKING to avoid runtime imports
if TYPE_CHECKING:
    from pyMAOS.frame2d import R2Frame
from pyMAOS.units_mod import ureg
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
        zero_force = 0 * ureg(Riy.units)

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

        from pyMAOS.units_mod import INTERNAL_MOMENT_UNIT, ureg
        self.c1 = ureg.Quantity(0, INTERNAL_MOMENT_UNIT)
        self.c2 = -1 * p * a
        self.c3 = (p * a * (a - (2 * L)) * (a - L)) / (6 * L)
        self.c4 = (p * a * ((a * a) + (2 * L * L))) / (6 * L)
        self.c5 = 0
        self.c6 = (-1 * p * a * a * a) / 6

        # Simple End Reactions
        self.Riy = self.p * ((self.a - self.L) / self.L)
        self.Rjy = -1 * self.p * self.a * (1 / self.L)
        print("Riy:", self.Riy, "Rjy:", self.Rjy)
        # Piecewise Functions
        # [co....cn x^n] [xa, xb]
        Vy = [
            [[self.Riy], [ureg.Quantity(0, self.a.units), self.a]],
            [[self.Riy + self.p], [self.a, self.L]],
        ]
        print("Vy:"); print_quantity_nested_list(Vy)
        Mz = [
            [[self.c1, self.Riy], [0 * ureg(self.a.units), self.a]],
            [[self.c2, self.Riy + self.p], [self.a, self.L]],
        ]
        print("Mz:"); print_quantity_nested_list(Mz)
        Sz = [
            [[self.c3, self.c1, self.Riy / 2], [0 * ureg(self.a.units), self.a]],
            [[self.c4, self.c2, (self.Riy + self.p) / 2], [self.a, self.L]],
        ]
        Sz[0][0] = [i / self.EI for i in Sz[0][0]]
        Sz[1][0] = [i / self.EI for i in Sz[1][0]]
        print("Sz:"); print_quantity_nested_list(Sz)
        Dy = [
            [[self.c5, self.c3, self.c1 / 2, self.Riy / 6], [0 * ureg(self.a.units), self.a]],
            [
                [self.c6, self.c4, self.c2 / 2, (self.Riy + self.p) / 6],
                [self.a, self.L],
            ],
        ]
        Dy[0][0] = [i / self.EI for i in Dy[0][0]]
        Dy[1][0] = [i / self.EI for i in Dy[1][0]]
        #print(Dy)
        print("Dy:")
        print_quantity_nested_list(Dy, precision=2, width=20)

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

    # def integration_constants(self):
    #     P = self.p
    #     a = self.a
    #     L = self.L
    #     import inspect;
    #     print(f"{inspect.getfile(inspect.currentframe())}:{inspect.currentframe().f_lineno}")
    #     self.c01 = 0
    #     self.c02 = -1 * P * a
    #     self.c03 = (P * a * (a - (2 * L)) * (a - L)) / (6 * L)
    #     self.c04 = (P * a * ((a * a) + (2 * L * L))) / (6 * L)
    #     self.c05 = 0
    #     self.c06 = (-1 * P * a * a * a) / 6

    def FEF(self):
        p = self.p
        a = self.a
        L = self.L

        Miz = -1 * (p * a * (a - L) * (a - L)) / (L * L)
        Mjz = -1 * (p * a * a * (a - L)) / (L * L)
        Riy = self.Riy + (Miz / L) + (Mjz / L)
        Rjy = self.Rjy - (Miz / L) - (Mjz / L)

        # Create zeros with the same units as Riy
        zero_force = 0 * ureg(Riy.units)

        return [zero_force, Riy, Miz, zero_force, Rjy, Mjz]