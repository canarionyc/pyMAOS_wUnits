import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def bernstein_shape_functions(order, x):
    """Generate Bernstein shape functions for a higher-order finite element."""
    n = order
    shape_funcs = []

    for i in range(n+1):
        # Bernstein polynomial of degree n
        bi = lambda t: special.comb(n, i) * t**i * (1-t)**(n-i)
        shape_funcs.append(bi(x))

    return np.array(shape_funcs)

# Generate shape functions for a cubic element
x = np.linspace(0, 1, 100)
shape_functions = bernstein_shape_functions(order=3, x=x)

# Plot the shape functions
plt.figure(figsize=(8, 5))
for i, sf in enumerate(shape_functions):
    plt.plot(x, sf, label=f'N_{i}')

plt.title('Bernstein Shape Functions for Higher-Order Finite Element')
plt.xlabel('Local coordinate ξ')
plt.ylabel('Shape function value')
plt.grid(True)
plt.legend()
plt.show()