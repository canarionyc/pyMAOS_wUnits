#%% Setup the problem using SymPy
import sympy as sp

#%% Define the problem parameters
# Define symbols
L, a, P = sp.symbols('L a P', positive=True)  # Length, load position, load magnitude
R_A, R_B = sp.symbols('R_A R_B')  # Vertical reactions at supports
M_A, M_B = sp.symbols('M_A M_B')  # Moments at supports

#%% Set up equilibrium equations
# For a beam with fixed ends and a point load P at distance a from the left end

# 1. Force equilibrium in vertical direction:
# Sum of all vertical forces must equal zero
force_eq = R_A + R_B - P

# 2. Moment equilibrium about point A (left end)
# Clockwise moments are positive
moment_eq_A = M_A + M_B + R_B*L - P*a

# 3. Moment equilibrium about point x=a (where load P is applied)
moment_eq_a = M_A + M_B + R_A*a - R_B*(L-a)

# 4. Moment equilibrium about point x=L (right end)
moment_eq_L = M_A + M_B - R_A*L + P*(L-a)

#%% Display the equilibrium equations
print("Equilibrium Equations:")
print(f"Force equilibrium: {force_eq} = 0")
print(f"Moment equilibrium at A: {moment_eq_A} = 0")
print(f"Moment equilibrium at x=a: {moment_eq_a} = 0")
print(f"Moment equilibrium at L: {moment_eq_L} = 0")

#%% Solve the equilibrium equations
# Now we have 4 equations and 4 unknowns (R_A, R_B, M_A, M_B)
eq_system = [
	force_eq,
	moment_eq_A,
	moment_eq_a,
	moment_eq_L
]

# Solve for all unknowns
solution = sp.solve(eq_system, [R_A, R_B, M_A, M_B])

# Extract the solutions
R_A_sol = solution[R_A]
R_B_sol = solution[R_B]
M_A_sol = solution[M_A]
M_B_sol = solution[M_B]

#%% Simplify and display results
R_A_sol = sp.simplify(R_A_sol)
R_B_sol = sp.simplify(R_B_sol)
M_A_sol = sp.simplify(M_A_sol)
M_B_sol = sp.simplify(M_B_sol)

print("\nSolution:")
print(f"Reaction at A: R_A = {R_A_sol}")
print(f"Reaction at B: R_B = {R_B_sol}")
print(f"Moment at A: M_A = {M_A_sol}")
print(f"Moment at B: M_B = {M_B_sol}")

#%% Verification of results
# Check if our solution satisfies the equilibrium equations
force_check = force_eq.subs([(R_A, R_A_sol), (R_B, R_B_sol)])
moment_check_A = moment_eq_A.subs([(R_A, R_A_sol), (R_B, R_B_sol), (M_A, M_A_sol), (M_B, M_B_sol)])
moment_check_a = moment_eq_a.subs([(R_A, R_A_sol), (R_B, R_B_sol), (M_A, M_A_sol), (M_B, M_B_sol)])
moment_check_L = moment_eq_L.subs([(R_A, R_A_sol), (R_B, R_B_sol), (M_A, M_A_sol), (M_B, M_B_sol)])

print("\nVerification:")
print(f"Force equilibrium check: {force_check} (should be 0)")
print(f"Moment equilibrium at A check: {sp.simplify(moment_check_A)} (should be 0)")
print(f"Moment equilibrium at a check: {sp.simplify(moment_check_a)} (should be 0)")
print(f"Moment equilibrium at L check: {sp.simplify(moment_check_L)} (should be 0)")

#%% Numerical example and plotting
import numpy as np
import matplotlib.pyplot as plt

# Define a specific example with numbers
L_val = 10  # Length of the beam
a_val = 4   # Load position
P_val = 100 # Load magnitude

# Calculate the reactions and moments
R_A_val = R_A_sol.subs({L: L_val, a: a_val, P: P_val})
R_B_val = R_B_sol.subs({L: L_val, a: a_val, P: P_val})
M_A_val = M_A_sol.subs({L: L_val, a: a_val, P: P_val})
M_B_val = M_B_sol.subs({L: L_val, a: a_val, P: P_val})

print("\nNumerical Solution:")
print(f"Reaction at A: R_A = {R_A_val}")
print(f"Reaction at B: R_B = {R_B_val}")
print(f"Moment at A: M_A = {M_A_val}")
print(f"Moment at B: M_B = {M_B_val}")

# Plot the shear and moment diagrams
x = np.linspace(0, L_val, 100)
shear_force = R_A_val - P_val * (x >= a_val)
bending_moment = M_A_val + R_A_val * x - P_val * (x - a_val) * (x >= a_val)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
ax1.plot(x, shear_force, label='Shear Force')
ax1.axhline(0, color='red', lw=0.8, ls='--')
ax1.set_ylabel('Shear Force')
ax1.set_title('Shear Force Diagram')
ax1.legend()

ax2.plot(x, bending_moment, label='Bending Moment', color='orange')
ax2.axhline(0, color='red', lw=0.8, ls='--')
ax2.set_xlabel('Position along the beam')
ax2.set_ylabel('Bending Moment')
ax2.set_title('Bending Moment Diagram')
ax2.legend()

plt.tight_layout()
plt.show()
