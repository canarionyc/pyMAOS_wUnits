
import math
import numpy as np

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


class Piecewise_Polynomial:
    def __init__(self, functions=[[[0], [0, 0]]]):
        self.functions = functions

    def __str__(self):
        out = ""
        for line in self.functions:
            func = ""
            for i, coeff in enumerate(line[0]):
                if i == 0:
                    func += f"{coeff:6.2f}"
                else:
                    func += f" + {coeff:6.2f} x^{i}"
            func += f"  for {line[1][0]:.2f} <= x <= {line[1][1]:.2f}\n"
            out += func
        return out
    def __repr__(self):
        return f"Piecewise_Polynomial(functions={self.functions})"

    def evaluate(self, x):
        """
        Given a piecewise function and an x evaluate the results
        """
        # in the context of the beam model a tolerance of 1E-6 will
        # yield acceptable results as we are evaluating normal polynomials
        tol = 0.000001
        
        # initialize res to avoid an ref before assignment error in
        # the case where the below reaches pass for all conditions.
        piece_function = self.functions
        res = 0

        if piece_function == []:
            res = 0
        else:
            for line in piece_function:
                if (line[1][0] - tol) < x <= (line[1][1] + tol):
                    res = polynomial_evaluation(line[0], x)
                else:
                    # x is not in the current functions range
                    pass
        return res

    def roots(self):
        """
        Given a piecewise function return a list
        of the location of zeros or sign change
        """
        piece_function = self.functions
        zero_loc = []
        i = 0
        for line in piece_function:
            if len(line[0]) == 1 and i == 0:
                pass  # If function is a value then there is no chance for a sign change
            else:
                a = polynomial_evaluation(
                    line[0], line[1][0] + 0.0001
                )  # value at start of bounds
                b = polynomial_evaluation(
                    line[0], line[1][1] - 0.0001
                )  # value at end of bounds

                if a == 0:
                    zero_loc.append(line[1][0])
                elif b == 0:
                    zero_loc.append(line[1][1])
                else:
                    # if signs are the the same a/b will result in a positive value
                    coeff = line[0][::-1]
                    c = np.roots(coeff)
                    # Some real solutions may contain a very small imaginary part
                    # account for this with a tolerance on the imaginary
                    # part of 1e-5
                    c = c.real[abs(c.imag) < 1e-5]
                    for root in c:
                        # We only want roots that are with the piece range
                        if line[1][0] < root <= line[1][1]:
                            zero_loc.append(root)
                        else:
                            pass
                if i == 0:
                    pass
                else:
                    # value at end of previous bounds
                    d = polynomial_evaluation(
                        piece_function[i - 1][0], line[1][0] - 0.0001
                    )

                    if d == 0:
                        pass
                    elif a / d < 0:
                        zero_loc.append(line[1][0])
                    else:
                        pass
            i += 1
        zero_loc = sorted(set(zero_loc))
        return zero_loc

    def combine(self, other, LF, LFother):
        """
        Join two piecewise functions to create one piecewise function ecompassing
        the ranges and polynomials associated with each
        """
        Fa = self.functions
        Fb = other.functions
        LFa = LF
        LFb = LFother

        functions = [Fa, Fb]
        LF = [LFa, LFb]

        # Gather the ranges for each piece of the the two input functions
        ab = []
        for func in Fa:
            ab.append(func[1][0])
            ab.append(func[1][1])
        for func in Fb:
            ab.append(func[1][0])
            ab.append(func[1][1])
        ab = list(set(ab))
        ab.sort()

        f_out = []

        for i, j in enumerate(ab):
            if i == 0:
                piece_range = [0, j]
            else:
                piece_range = [ab[i - 1], j]
            if piece_range == [0, 0]:
                pass
            else:
                f = []

                for i, func in enumerate(functions):
                    for piece in func:
                        if (
                            piece[1][0] < piece_range[1]
                            and piece[1][1] >= piece_range[1]
                        ):
                            # difference in number of coefficients
                            eq_len_delta = len(piece[0]) - len(f)

                            if eq_len_delta > 0:
                                f.extend([0] * eq_len_delta)
                            elif eq_len_delta < 0:
                                piece[0].extend([0] * abs(eq_len_delta))
                            else:
                                pass
                            f = [j * LF[i] + k for j, k in zip(piece[0], f)]
                        else:
                            pass
                f_out.append([f, piece_range])
        return Piecewise_Polynomial(f_out)