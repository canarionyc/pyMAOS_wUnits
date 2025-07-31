import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def bernstein_basis(t, i, n):
    """Calculate the Bernstein polynomial basis."""
    return special.comb(n, i) * t**i * (1-t)**(n-i)

def beam_shape(x, coeffs):
    """Define beam height along its length using Bernstein polynomials."""
    n = len(coeffs) - 1
    height = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        height += coeff * bernstein_basis(x, i, n)
    return height

def objective_function(coeffs):
    """Objective function: minimize material volume while maintaining stiffness."""
    # Use smoother penalty for height constraints
    min_height = 0.05
    height_penalty = 0
    for coeff in coeffs:
        if coeff < min_height:
            height_penalty += 100 * (min_height - coeff)**2

    # Calculate beam volume
    x = np.linspace(0, 1, 100)
    heights = beam_shape(x, coeffs)
    volume = np.trapz(heights, x)

    # Safety check for minimum height to avoid division by zero
    min_actual_height = np.min(heights)
    if min_actual_height < 0.01:
        stiffness_penalty = 1000  # Large but finite penalty
    else:
        stiffness_penalty = 1.0 / min_actual_height**3

    print(f"DEBUG: Volume={volume:.4f}, Min height={min_actual_height:.4f}, Penalty={stiffness_penalty:.4f}")

    return volume + 0.01 * stiffness_penalty + height_penalty

# Add bounds to prevent negative or very small values
initial_coeffs = [0.1, 0.15, 0.15, 0.1]
bounds = [(0.05, 0.3) for _ in range(len(initial_coeffs))]

# Optimize with bounds
result = minimize(objective_function, initial_coeffs, method='L-BFGS-B', bounds=bounds)
optimized_coeffs = result.x

# Plot results
x = np.linspace(0, 1, 100)
initial_shape = beam_shape(x, initial_coeffs)
optimized_shape = beam_shape(x, optimized_coeffs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot shapes
ax1.plot(x, initial_shape, 'b--', label='Initial shape')
ax1.plot(x, optimized_shape, 'r-', label='Optimized shape')
ax1.set_title('Beam Shape Optimization using Bernstein Polynomials')
ax1.set_xlabel('Normalized beam length')
ax1.set_ylabel('Beam height')
ax1.grid(True)
ax1.legend()

# Plot control points
control_x = np.linspace(0, 1, len(optimized_coeffs))
ax2.stem(control_x, optimized_coeffs, 'g-', label='Control points')
ax2.plot(x, optimized_shape, 'r-', alpha=0.5)
ax2.set_title('Control Points of Optimized Shape')
ax2.set_xlabel('Normalized beam length')
ax2.set_ylabel('Control point value')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Initial coefficients: {initial_coeffs}")
print(f"Optimized coefficients: {optimized_coeffs}")
print(f"Optimization status: {result.success}, message: {result.message}")
