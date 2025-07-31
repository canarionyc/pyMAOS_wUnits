import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def bernstein_poly(i, n, t):
    """Bernstein polynomial basis function."""
    return special.comb(n, i) * t**i * (1-t)**(n-i)

def airfoil_shape(x, coeffs):
    """Generate airfoil profile using Bernstein polynomials."""
    n = len(coeffs) - 1
    y = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        y += coeff * bernstein_poly(i, n, x)
    return y

# Define control points for upper airfoil surface
coeffs_upper = [0.0, 0.05, 0.08, 0.06, 0.03, 0.0]  # NACA-like shape

# Generate airfoil coordinates
x = np.linspace(0, 1, 100)
y_upper = airfoil_shape(x, coeffs_upper)

plt.figure(figsize=(10, 4))
plt.plot(x, y_upper, 'b-')
plt.plot(x, -y_upper*0.7, 'b-')  # Simple lower surface
plt.axis('equal')
plt.grid(True)
plt.title('Airfoil Shape Parameterized with Bernstein Polynomials')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.show()