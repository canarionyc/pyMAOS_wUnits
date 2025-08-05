import os
import yaml
import numpy as np
from pyMAOS.pymaos_sections import Section
from pyMAOS.pymaos_units import set_unit_system, IMPERIAL_UNITS

print(os.getcwd())


# Set the unit system to imperial
set_unit_system(IMPERIAL_UNITS, "imperial")

# Load sections from the YAML file
sections_yml = os.path.join("sections.yml")
with open(sections_yml, 'r') as file:
    # Use unsafe_load to allow object instantiation
    sections_list = yaml.unsafe_load(file)

# Display section properties
for section in sections_list:
    print(section)  # This will use the __str__ method of Section

    # Print Area properties
    print(f"Area value: {section.Area}")
    print(f"Area with display units: {section.Area.to(IMPERIAL_UNITS['area'])}")

    # Print Ixx properties with NaN handling
    print(f"Ixx value: {section.Ixx}")
    if not np.isnan(section.Ixx):
        print(f"Ixx with display units: {section.Ixx.to(IMPERIAL_UNITS['moment_of_inertia'])}")

    # Print Iyy properties with NaN handling
    print(f"Iyy value: {section.Iyy}")
    if not np.isnan(section.Iyy):
        print(f"Iyy with display units: {section.Iyy.to(IMPERIAL_UNITS['moment_of_inertia'])}")

    print("-" * 50)