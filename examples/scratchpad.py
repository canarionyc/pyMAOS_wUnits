
import pint
ureg = pint.UnitRegistry()
q = pint.Quantity(1, '1 / meter ** 4 / pascal')

# Simplify to base units
q_base = q.to_base_units()
print("Base units:", q_base)

# Simplify to reduced units (combines and cancels units)
q_reduced = q.to_reduced_units()
print("Reduced units:", q_reduced)