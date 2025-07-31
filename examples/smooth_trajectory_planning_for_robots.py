import numpy as np
from scipy.interpolate import BPoly
import matplotlib.pyplot as plt

# Define control points for a robot path
control_points = np.array([
    [0.0, 0.0],   # Start position
    [0.3, 0.5],   # Via point
    [0.7, 0.3],   # Via point
    [1.0, 0.8]    # End position
])

# Create a Bernstein polynomial representation of the path
x = np.linspace(0, 1, 100)
bp = BPoly.from_derivatives(
    xi=[0, 1],  # Normalized time interval [0, 1]
    yi=np.array([[control_points[0], control_points[3]-control_points[0]],  # Position and velocity at t=0
                [control_points[3], control_points[3]-control_points[2]]])  # Position and velocity at t=1
)

# Evaluate trajectory at specified times
times = np.linspace(0, 1, 100)
trajectory = bp(times)

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control points')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Smooth trajectory')
plt.grid(True)
plt.title('Robot Trajectory Planning using Bernstein Polynomials')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.legend()
plt.show()

