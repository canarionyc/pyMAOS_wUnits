import numpy as np
from scipy.interpolate import PPoly

# Define breakpoints (intervals)
breaks = [0, 1, 2]  # Breakpoints for the intervals [0, 1] and [1, 2]

# Define coefficients for each interval
# Coefficients are in descending order of powers (highest degree first)
coeffs = [
    [1, 0, 0],  # Coefficients for x^2 (interval [0, 1])
    [2, 1]      # Coefficients for 2x + 1 (interval [1, 2])
]

# Convert coefficients to a 2D array with shape (degree + 1, number of intervals)
max_degree = max(len(c) for c in coeffs)
c = np.zeros((max_degree, len(breaks) - 1))

for i, poly_coeffs in enumerate(coeffs):
    c[-len(poly_coeffs):, i] = poly_coeffs[::-1]  # Reverse order for descending powers

# Create the PPoly object
ppoly = PPoly(c, breaks)

# Evaluate the piecewise polynomial at specific points
x_values = np.linspace(0, 2, 100)
y_values = ppoly(x_values)

# Debug print statements
print(f"Breakpoints: {breaks}")
print(f"Coefficient matrix:\n{c}")
print(f"Evaluated values:\n{y_values}")