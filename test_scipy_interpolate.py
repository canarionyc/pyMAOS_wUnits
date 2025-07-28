import numpy as np
from scipy.interpolate import PPoly

# Define breakpoints (x values where segments meet)
x = [0, 1, 2]

# In PPoly, coefficients are arranged with the highest degree term first, and each column represents a different segment of the piecewise function.
# Define coefficients for each segment
# Shape (3, 2) where:
# - 3 rows for coefficients (highest power first)
# - 2 columns for segments
c = np.array([
    [1, 4],    # Coefficients of x² term for segments 0 and 1
    [2, 5],    # Coefficients of x term for segments 0 and 1
    [3, 6]     # Coefficients of constant term for segments 0 and 1
])

# Create the piecewise polynomial
pp = PPoly(c, x)

# Evaluate, differentiate, integrate
print(0.5**2+2*0.5+3)  ; print(pp(0.5)) # Evaluate at x=0.5;
# Evaluate at x=0.5
print(2*0.5**1+2) ; print(pp.derivative()(0.5))  # Evaluate derivative at x=0.5
pp.antiderivative()(0.5)  # Evaluate integral at x=0.5