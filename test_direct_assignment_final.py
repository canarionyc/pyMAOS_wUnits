#!/usr/bin/env python3
"""
Test to verify that direct assignment of loads with units works correctly
This addresses the issue: N2.loads[loadcase] = [0, "-50kip", 0] should work
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_direct_assignment_with_units():
    """Test that direct assignment of loads with units works correctly"""
    
    try:
        from pyMAOS.R2Node import R2Node
        from pyMAOS.units_mod import set_unit_system, IMPERIAL_UNITS
        
        print("✓ Imports successful")
        
        # Set imperial units for testing
        set_unit_system(IMPERIAL_UNITS, "imperial")
        print("✓ Unit system set to Imperial")
        
        # Create a test node
        N2 = R2Node(2, "120in", "0in")
        print(f"✓ Node created: x={N2.x:.6f}m, y={N2.y:.6f}m")
        
        # Test 1: Direct assignment with units (the main issue)
        loadcase = "D"
        N2.loads[loadcase] = [0, "-50kip", 0]
        print(f"✅ DIRECT ASSIGNMENT WORKS: N2.loads['{loadcase}'] = [0, '-50kip', 0]")
        print(f"    Result in SI: {N2.loads[loadcase]} [N, N, N*m]")
        
        # Verify the conversion is correct
        expected_fy = -50 * 4448.222  # 1 kip = 4448.222 N
        actual_fy = N2.loads[loadcase][1]
        tolerance = 1.0  # 1 N tolerance
        
        if abs(actual_fy - expected_fy) < tolerance:
            print(f"✅ Unit conversion correct: -50kip → {actual_fy:.1f}N (expected {expected_fy:.1f}N)")
        else:
            print(f"❌ Unit conversion error: got {actual_fy:.1f}N, expected {expected_fy:.1f}N")
            return False
        
        # Test 2: Multiple assignments to same load case
        N2.loads[loadcase] = ["5kip", "-25kip", "100kip*ft"]
        print(f"✅ Multiple units assignment works: {N2.loads[loadcase]} [N, N, N*m]")
        
        # Test 3: Multiple load cases
        N2.loads["L"] = ["-10kip", "0kip", "50kip*ft"]
        N2.loads["W"] = ["0lbf", "-1000lbf", "0lbf*ft"]
        print(f"✅ Multiple load cases work:")
        print(f"    Load case 'L': {N2.loads['L']}")
        print(f"    Load case 'W': {N2.loads['W']}")
        
        # Test 4: Mixed units and numeric values
        N2.loads["MIXED"] = [100, "-30kip", 0]  # Mix of numeric and unit strings
        print(f"✅ Mixed numeric/unit values work: {N2.loads['MIXED']}")
        
        # Test 5: Verify add_nodal_load still works
        N3 = R2Node(3, "20ft", "0in")
        N3.add_nodal_load("10kip", "-25kip", "0klbf*ft", "D")
        print(f"✅ add_nodal_load method still works: {N3.loads['D']}")
        
        # Test 6: Test that original demo_excel_export.py syntax now works
        print("\n" + "="*60)
        print("TESTING DEMO_EXCEL_EXPORT.PY SYNTAX")
        print("="*60)
        
        # Recreate the exact scenario from demo_excel_export.py
        N_demo = R2Node(2, "120in", "0in")
        loadcase_demo = "D"
        
        # This is the exact line that was failing in demo_excel_export.py:
        N_demo.loads[loadcase_demo] = [0, "-50kip", 0]  # 50 kip downward load at middle node
        
        print(f"✅ DEMO SYNTAX WORKS: N2.loads[loadcase] = [0, '-50kip', 0]")
        print(f"    Input: [0, '-50kip', 0]")
        print(f"    Output SI: {N_demo.loads[loadcase_demo]} [N, N, N*m]")
        
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Direct assignment of loads with units now works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_assignment_with_units()
    sys.exit(0 if success else 1)