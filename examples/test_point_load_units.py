#!/usr/bin/env python3
"""
Test script to verify that add_point_load accepts strings with units
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyMAOS.node2d import R2Node
from pyMAOS.elements import R2Frame
from pyMAOS.pymaos_linear_elastic_material import LinearElasticMaterial as Material
from pyMAOS.pymaos_sections import Section
from pyMAOS.pymaos_units import set_unit_system, IMPERIAL_UNITS

def test_point_load_with_units():
    """Test add_point_load with string inputs containing units"""
    
    # Set imperial units for demo
    set_unit_system(IMPERIAL_UNITS, "imperial")
    
    # Create simple beam nodes
    N1 = R2Node(1, "0ft", "0ft")
    N2 = R2Node(2, "10ft", "0ft")
    
    # Apply restraints
    N1.restrainAll()
    N2.restrainTranslation()
    
    # Create material and section
    steel = Material(uid=1, density="0.490klb/ft^3", E="29000ksi", nu=0.3)
    beam_section = Section(uid=1, Area="11.8in^2", Ixx="518in^4")
    
    # Create frame element
    beam = R2Frame(1, N1, N2, steel, beam_section)
    
    # Test point load with string units
    print("Testing add_point_load with string units...")
    
    # Test 1: Point load with imperial units
    beam.add_point_load("-25kip", "5ft", case="D", direction="y")
    print("✓ Successfully added point load: -25kip at 5ft")
    
    # Test 2: Point load with mixed units
    beam.add_point_load("-10klbf", "60in", case="L", direction="y")
    print("✓ Successfully added point load: -10klbf at 60in")
    
    # Test 3: Point load with percentage position
    beam.add_point_load("-15kip", 50, case="D", direction="y", location_percent=True)
    print("✓ Successfully added point load: -15kip at 50% of span")
    
    # Test 4: Axial load with units
    beam.add_point_load("5kip", "3ft", case="D", direction="xx")
    print("✓ Successfully added axial load: 5kip at 3ft")
    
    # Test 5: Global Y direction load
    beam.add_point_load("-20kip", "7.5ft", case="D", direction="Y")
    print("✓ Successfully added global Y load: -20kip at 7.5ft")
    
    # Test 6: Add moment load with units
    beam.add_moment_load("100kip*ft", "4ft", case="D")
    print("✓ Successfully added moment load: 100kip*ft at 4ft")
    
    # Test 7: Distributed load with units
    beam.add_distributed_load("-0.5kip/in", "-0.5kip/in", "0ft", "10ft", case="D")
    print("✓ Successfully added distributed load: -0.5kip/in uniform")
    
    print("\nAll tests passed! ✅")
    print(f"Frame element has {len(beam.loads)} loads applied")
    
    return beam

if __name__ == "__main__":
    test_point_load_with_units()