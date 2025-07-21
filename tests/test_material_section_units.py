#!/usr/bin/env python3
"""
Test script to demonstrate Material and Section parsing with units
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
from pyMAOS.units_mod import set_unit_system, IMPERIAL_UNITS, SI_UNITS

def test_material_and_section_units():
    """Test Material and Section creation with various unit formats"""
    
    print("Testing Material and Section unit parsing...")
    print("=" * 60)
    
    # Set imperial unit system for display
    set_unit_system(IMPERIAL_UNITS, "imperial")
    print(f"Using Imperial units: {IMPERIAL_UNITS}\n")
    
    print("=== MATERIAL TESTS ===")
    print("-" * 30)
    
    # Test case 1: Material with imperial units
    print("Test 1: Steel material with imperial units")
    steel_imperial = Material(uid=1, density="0.490klb/ft^3", E="29000ksi", nu=0.3)
    print(f"Result: {steel_imperial}")
    print(f"Internal (SI): density={steel_imperial.density:.6f} kg/m³, E={steel_imperial.E:.0f} Pa\n")
    
    # Test case 2: Material with mixed units
    print("Test 2: Material with mixed units")
    mixed_material = Material(uid=2, density="2400kg/m^3", E="200000MPa", nu=0.25)
    print(f"Result: {mixed_material}")
    print(f"Internal (SI): density={mixed_material.density:.6f} kg/m³, E={mixed_material.E:.0f} Pa\n")
    
    # Test case 3: Material with numeric values (SI assumed)
    print("Test 3: Material with numeric values (SI units assumed)")
    numeric_material = Material(uid=3, density=7850, E=200e9, nu=0.3)
    print(f"Result: {numeric_material}")
    print(f"Internal (SI): density={numeric_material.density:.6f} kg/m³, E={numeric_material.E:.0f} Pa\n")
    
    print("=== SECTION TESTS ===")
    print("-" * 30)
    
    # Test case 4: Section with imperial units
    print("Test 4: W-section with imperial units")
    w_section = Section(uid=1, Area="11.8in^2", Ixx="518in^4")
    print(f"Result: {w_section}")
    print(f"Internal (SI): Area={w_section.Area:.8f} m², Ixx={w_section.Ixx:.12f} m⁴\n")
    
    # Test case 5: Section with metric units
    print("Test 5: Section with metric units")
    metric_section = Section(uid=2, Area="7600mm^2", Ixx="215.6e6mm^4")
    print(f"Result: {metric_section}")
    print(f"Internal (SI): Area={metric_section.Area:.8f} m², Ixx={metric_section.Ixx:.12f} m⁴\n")
    
    # Test case 6: Section with numeric values (SI assumed)
    print("Test 6: Section with numeric values (SI units assumed)")
    numeric_section = Section(uid=3, Area=0.0076, Ixx=0.0002156)
    print(f"Result: {numeric_section}")
    print(f"Internal (SI): Area={numeric_section.Area:.8f} m², Ixx={numeric_section.Ixx:.12f} m⁴\n")
    
    print("=== UNIT CONVERSION VERIFICATION ===")
    print("-" * 40)
    print("Imperial to SI conversions:")
    print("11.8 in² = 11.8 × (0.0254)² = 0.00761290 m²")
    print("518 in⁴ = 518 × (0.0254)⁴ = 0.000215658 m⁴")
    print("29000 ksi = 29000 × 6,894,757 = 199,947,953,000 Pa")
    print("0.490 klb/ft³ = 0.490 × 16,018.5 = 7,849 kg/m³")
    
    # Test switching unit systems
    print("\n=== UNIT SYSTEM SWITCHING ===")
    print("-" * 40)
    print("Switching to SI units for display...")
    set_unit_system(SI_UNITS, "SI")
    
    print(f"Steel material (now in SI): {steel_imperial}")
    print(f"W-section (now in SI): {w_section}")

if __name__ == "__main__":
    test_material_and_section_units()