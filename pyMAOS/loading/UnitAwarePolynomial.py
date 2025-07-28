from pint import Quantity
from numpy.polynomial import Polynomial

from pyMAOS.units_mod import UnitManager, ureg


class UnitAwarePolynomial(Polynomial):
    def __init__(self, coeffs_with_units, y_units=None,x_units=None):
        # Store the unit information
        self.coeffs_with_units = list(coeffs_with_units)
        c0 = coeffs_with_units[0]
        self.y_units = c0.to_base_units().units if isinstance(c0, Quantity) else ureg.dimensionless
        self.y_units = y_units or self.y_units
        self.x_units = x_units or ureg.dimensionless
        # Initialize the base class with magnitude values
        magnitudes = [c.magnitude if isinstance(c, Quantity) else c for c in self.coeffs_with_units]
        super().__init__(magnitudes)

        # Debug info
        print(f"Created UnitAwarePolynomial with y unit: {self.y_units} x unit: {self.x_units}")

    def __call__(self, x):
        """
        Evaluate polynomial at points x (works with scalar or array input).

        Parameters
        ----------
        x : scalar, array_like, or Quantity
            Points at which to evaluate the polynomial

        Returns
        -------
        scalar or ndarray with units
            Polynomial values at x with appropriate units
        """
        # Handle unit conversion if x has units
        if isinstance(x, Quantity):
            # Check unit compatibility
            if x.dimensionality != self.x_units.dimensionality:
                raise ValueError(f"Input x units {x.units} not compatible with polynomial x units {self.x_units}")
            # Extract magnitude in the polynomial's x units
            x_magnitude = x.to(self.x_units).magnitude
        else:
            # No units provided - use as is
            x_magnitude = x

        # Use base class for calculation (automatically handles arrays)
        result_magnitude = super().__call__(x_magnitude)

        # Apply units to the result (works for both scalar and array results)
        return result_magnitude * self.y_units

    def __add__(self, other):
        # Handle unit-aware addition
        if isinstance(other, UnitAwarePolynomial):
            if self.y_units != other.y_units or self.x_units != other.x_units:
                raise ValueError(f"Cannot add polynomials with different y units: {self.y_units} and {other.y_units} or x units: {self.x_units} and {other.x_units}")
            result = super().__add__(other)
            # Create new UnitAwarePolynomial with the result
            return UnitAwarePolynomial([c * self.y_units for c in result.coef], y_units=self.y_units, x_units=self.x_units)
          
        return NotImplemented

    def __mul__(self, other):
        # Handle multiplication with scalar or other polynomial
        result = super().__mul__(other)
        if isinstance(other, (int, float)):
            # Simple scalar multiplication
            return UnitAwarePolynomial
        elif isinstance(other, Quantity):
            # Multiply by a quantity
            new_unit = self.y_units * other.units
            return UnitAwarePolynomial([c * new_unit for c in result.coef])
        return NotImplemented

    def deriv(self, m=1):
        # Override derivative to maintain unit awareness
        result = super().deriv(m)
        # Calculate the new unit (e.g., meters → meters/second)
        new_y_units = self.y_units / self.x_units**m
        return UnitAwarePolynomial([c * new_y_units for c in result.coef], y_units=new_unit, x_units=self.x_units)

    def integ(self, m=1, k=0):
        # Override integration to maintain unit awareness
        result = super().integ(m, k)
        # Calculate the new unit (e.g., meters/second → meters)
        new_y_units = self.y_units * self.x_units**m
        return UnitAwarePolynomial([c * new_y_units for c in result.coef], y_units=new_y_units, x_units=self.x_units)