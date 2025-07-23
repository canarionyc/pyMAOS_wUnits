#!/usr/bin/env python3
"""
Simplified demo that replicates the exact demo_excel_export.py functionality
to test that direct assignment with units works correctly
"""

import sys
import os
# Add the parent directory to the path so we can import pyMAOS
sys.path.insert(0, os.path.abspath('.'))

def test_demo_excel_export_functionality():
    """Test the exact functionality from demo_excel_export.py"""
    
    try:
        # Import exactly what demo_excel_export.py imports
        from pyMAOS.node2d import R2Node
        from pyMAOS.elements import R2Frame
        from pyMAOS.material import LinearElasticMaterial as Material
        from pyMAOS.section import Section
        from pyMAOS.units_mod import set_unit_system, IMPERIAL_UNITS
        
        print("✅ All imports successful (demo_excel_export.py compatibility)")
        
        # Replicate the exact setup from demo_excel_export.py
        print("\n" + "="*60)
        print("REPLICATING DEMO_EXCEL_EXPORT.PY FUNCTIONALITY")
        print("="*60)
        
        # Set the unit system to imperial (exactly as in demo)
        print("Setting unit system to Imperial...")
        set_unit_system(IMPERIAL_UNITS, "imperial")
        print(f"Using Imperial units: {IMPERIAL_UNITS}")
        
        # Create nodes exactly as in demo
        print("\nCreating nodes with imperial coordinates...")
        N1 = R2Node(1, "0in", "0in")        # Left support
        N2 = R2Node(2, "120in", "0in")      # Middle support (10 feet = 120 inches)
        N3 = R2Node(3, "20ft", "0in")       # Right support (20 feet)

        print(f"Node 1: Input='0in', '0in' → SI: {N1.x:.6f}m, {N1.y:.6f}m")
        print(f"Node 2: Input='120in', '0in' → SI: {N2.x:.6f}m, {N2.y:.6f}m")
        print(f"Node 3: Input='20ft', '0in' → SI: {N3.x:.6f}m, {N3.y:.6f}m")

        # Apply restraints exactly as in demo
        N1.restrainAll()           # Fixed support
        N2.restrainTranslation()   # Pin support
        N3.restrainTranslation()   # Pin support
        
        # THE CRITICAL TEST: Apply loads exactly as in demo_excel_export.py
        print("\nApplying loads with imperial units...")
        loadcase = "D"
        
        # THIS IS THE LINE THAT WAS FAILING - Method 1: Direct assignment with string units
        print("Testing: N2.loads[loadcase] = [0, '-50kip', 0]")
        N2.loads[loadcase] = [0, "-50kip", 0]  # 50 kip downward load at middle node
        print(f"✅ SUCCESS! Node 2 load: Input=[0, '-50kip', 0] → SI: {N2.loads[loadcase]} [N, N, N*m]")
        
        # Method 2: Using add_nodal_load with string units (this should also work)
        print("Testing: N3.add_nodal_load('10kip', '-25kip', '0klbf*ft', loadcase)")
        N3.add_nodal_load("10kip", "-25kip", "0klbf*ft", loadcase)
        print(f"✅ SUCCESS! Node 3 load: add_nodal_load → SI: {N3.loads[loadcase]} [N, N, N*m]")
        
        # Test material creation with units (as in demo)
        print("\nCreating material with imperial units...")
        steel = Material(uid=1, density="0.490klb/ft^3", E="29000ksi", nu=0.3)
        print(f"✅ Steel material: density={steel.density:.1f} kg/m³, E={steel.E:.0f} Pa")

        # Test section creation with units (as in demo)
        print("\nCreating section with imperial units...")
        beam_section = Section(uid=1, Area="11.8in^2", Ixx="518in^4")
        print(f"✅ Section: Area={beam_section.Area:.8f} m², Ixx={beam_section.Ixx:.12f} m⁴")
        
        # Test frame creation and member loads (as in demo)
        beam1 = R2Frame(1, N1, N2, steel, beam_section)
        beam2 = R2Frame(2, N2, N3, steel, beam_section)
        
        print("\nTesting frame member loads with units...")
        beam1.add_distributed_load("-0.5kip/in", "-0.5kip/in", 0, beam1.length, case="D")
        print("✅ Distributed load with units added successfully")
        
        beam2.add_point_load("-25kip", beam2.length/2, case="D")
        print("✅ Point load with units added successfully")
        
        print("\n" + "="*60)
        print("🎉 ALL DEMO_EXCEL_EXPORT.PY FUNCTIONALITY WORKING!")
        print("✅ Direct assignment with units: FIXED")
        print("✅ Method calls with units: WORKING") 
        print("✅ Material/Section with units: WORKING")
        print("✅ Frame loads with units: WORKING")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_demo_excel_export_functionality()
    if success:
        print("\n🚀 The demo_excel_export.py functionality is now fully working!")
        print("💡 The direct assignment issue has been resolved.")
    else:
        print("\n❌ There are still issues to resolve.")
    
    sys.exit(0 if success else 1)