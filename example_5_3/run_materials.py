import os
import yaml
from pyMAOS.material import LinearElasticMaterial
from pyMAOS.units_mod import set_unit_system, IMPERIAL_UNITS

print(os.getcwd())


set_unit_system(IMPERIAL_UNITS, "imperial")  # Set the unit system to imperial

materials_yml = os.path.join("materials.yml")
with open(materials_yml, 'r') as file:
    # Use unsafe_load to allow object instantiation
    materials_list = yaml.unsafe_load(file)

for material_obj in materials_list:
    # Create LinearElasticMaterial objects from the YAML data
    print(material_obj)
    print(f"E value: {material_obj.E}")
    print(f"E with display units: {material_obj.E.to(IMPERIAL_UNITS['pressure'])}")
    print(f"Material {material_obj.uid} initialized with E={material_obj.E}, density={material_obj.density}, nu={material_obj.nu}")


