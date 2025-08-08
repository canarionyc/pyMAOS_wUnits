import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

# Create two polynomial segments with specified domains
# p1(x) = 1 + 2x for x in [0, 1]
p1 = Polynomial([1, 2], domain=[0, 1])

print("p1 representation:", p1)
print("p1(0) =", p1(0))
print("p1(1) =", p1(1))

# Create the same polynomial without specifying domain
p1_no_domain = Polynomial([1, 2])
print("\nWithout domain:")
print("p1_no_domain(0) =", p1_no_domain(0))


# p2(x) = 3 + 4x + x² for x in [1, 2]
p2 = Polynomial([3, 4, 1], domain=[1, 2])

# Function to evaluate the piecewise polynomial
def evaluate_piecewise(x):
    if 0 <= x <= 1:
        return p1(x)
    elif 1 < x <= 2:
        return p2(x)
    else:
        return np.nan  # Outside domain

# Test evaluation at various points
print("p1 at x=0.5:", p1(0.5))  # Within p1's domain
print("p2 at x=1.5:", p2(1.5))  # Within p2's domain

# What happens at domain boundaries?
print("p1 at domain boundary (x=1):", p1(1))
print("p2 at domain boundary (x=1):", p2(1))

# What happens outside the domain?
print("p1 evaluated outside domain (x=1.5):", p1(1.5))  # Still works but may not be what you expect

# Visualization
x_vals = np.linspace(0, 2, 100)
y_vals = [evaluate_piecewise(x) for x in x_vals]

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals)
plt.grid(True)
plt.title("Piecewise Polynomial Function")
plt.xlabel("x")
plt.ylabel("y")
plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)  # Mark the boundary
plt.savefig("piecewise_polynomial.png")  # Optional: save the plot