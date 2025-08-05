# Example usage of the new export_results_to_excel method

import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyMAOS.node2d import R2Node
from pyMAOS.elements import R2Frame
from pyMAOS.pymaos_linear_elastic_material import LinearElasticMaterial as Material
from pyMAOS.pymaos_sections import Section
import pyMAOS.structure2d as R2Struct
from pyMAOS.loadcombos import LoadCombo

# Import unit system utilities
from pyMAOS.pymaos_units import set_unit_system, IMPERIAL_UNITS

# Create a simple 2-span beam example
def demo_excel_export():
    """
    Demonstrates the new export_results_to_excel method with a simple beam structure
    using imperial units (kip, inch, psi) and the new R2Node coordinate/load parsing with units
    """
    
    # Set the unit system to imperial
    print("Setting unit system to Imperial...")
    set_unit_system(IMPERIAL_UNITS, "imperial")
    print(f"Using Imperial units: {IMPERIAL_UNITS}")
    
    # Create nodes using string coordinates with units - these will be automatically converted to SI
    print("\nCreating nodes with imperial coordinates (will be converted to SI internally)...")
    N1 = R2Node(1, "0in", "0in")        # Left support - using string coordinates with units
    N2 = R2Node(2, "120in", "0in")      # Middle support (10 feet = 120 inches)
    N3 = R2Node(3, "20ft", "0in")       # Right support (20 feet, mixed units demo)

    # Print the actual coordinates to show they were converted to SI
    print(f"Node 1: Input='0in', '0in' → SI: {N1.x:.6f}m, {N1.y:.6f}m")
    print(f"Node 2: Input='120in', '0in' → SI: {N2.x:.6f}m, {N2.y:.6f}m")
    print(f"Node 3: Input='20ft', '0in' → SI: {N3.x:.6f}m, {N3.y:.6f}m")

    # Apply restraints
    N1.restrainAll()           # Fixed support
    N2.restrainTranslation()   # Pin support
    N3.restrainTranslation()   # Pin support
    
    # Node list
    nodes = [N1, N2, N3]
    
    # Apply loads using string format with units - these will be automatically converted to SI
    print("\nApplying loads with imperial units (will be converted to SI internally)...")
    loadcase = "D"
    
    # Method 1: Direct assignment with string units
    N2.loads[loadcase] = [0, "-50kip", 0]  # 50 kip downward load at middle node
    print(f"Node 2 load: Input=[0, '-50kip', 0] → SI: {format_load_list(N2.loads[loadcase])} [N, N, N*m]")
    
    # Method 2: Using add_nodal_load with string units
    N3.add_nodal_load("10kip", "-25kip", "0klbf*ft", loadcase)
    print(f"Node 3 load: Input=add_nodal_load('10kip', '-25kip', '0klbf*ft') → SI: {N3.loads[loadcase]} [N, N, N*m]")
    
    # Create material and section (imperial units)
    # Steel: E = 29,000 ksi, density = 0.490 klb/ft³ = 0.490/1728 klb/in³ ≈ 0.000284 klb/in³
    print("\nCreating material with imperial units (will be converted to SI internally)...")
    steel = Material(uid=1, density="0.490klb/ft^3", E="29000ksi", nu=0.3)  # Steel material in imperial units
    print(f"Steel: Input=density='0.490klb/ft^3', E='29000ksi' → SI: density={steel.density:.1f} kg/m³, E={steel.E:.0f} Pa")

    # Typical W-section: Area = 11.8 in², Ixx = 518 in⁴
    print("\nCreating section with imperial units (will be converted to SI internally)...")
    beam_section = Section(uid=1, Area="11.8in^2", Ixx="518in^4")      # Typical beam section in imperial units
    print(f"Section: Input=Area='11.8in^2', Ixx='518in^4' → SI: Area={beam_section.Area:.8f} m², Ixx={beam_section.Ixx:.12f} m⁴")
    
    # Create frame members
    beam1 = R2Frame(1, N1, N2, steel, beam_section)
    beam2 = R2Frame(2, N2, N3, steel, beam_section)  

    # Member list
    members = [beam1, beam2]
    
    # Add some member loads for demonstration (imperial units)
    beam1.add_distributed_load("-0.5kip/in", "-0.5kip/in", 0, beam1.length, case="D")  # 0.5 kip/in uniform load on first span
    beam2.add_point_load("-25kip", beam2.length/2, case="D")                # 25 kip point load at midspan of second span

    # Create structure and pass the imperial units
    structure = R2Struct.R2Structure(nodes, members, units=IMPERIAL_UNITS)
    # structure.set_node_uids()
    # structure.set_member_uids()
    
    # Create load combination
    load_combo = LoadCombo("Dead Load", {"D": 1.0}, ["D"], False, "SLS")
    
    # Solve the structure
    print("\nSolving structure...")
    structure.solve_linear_static(load_combo, verbose=True)
    
    # Export results to Excel
    print("\nExporting results to Excel...")
    excel_file = structure.export_results_to_excel(
        load_combo, 
        output_file="beam_analysis_results_imperial.xlsx",
        include_visualization=True
    )
    
    print(f"Excel export completed: {excel_file}")
    
    # You can also export without visualization
    excel_file_no_viz = structure.export_results_to_excel(
        load_combo,
        output_file="beam_analysis_results_imperial_no_viz.xlsx", 
        include_visualization=False
    )
    
    print(f"Excel export (no visualization) completed: {excel_file_no_viz}")
    
    # Print summary of results in imperial units
    print("\n" + "="*60)
    print("ANALYSIS RESULTS (Imperial Units)")
    print("="*60)
    print(f"Unit System: Force={IMPERIAL_UNITS['force']}, Length={IMPERIAL_UNITS['length']}, Pressure={IMPERIAL_UNITS['pressure']}")
    print()
    print("Node Coordinates (converted to SI for internal calculations):")
    for node in nodes:
        print(f"  Node {node.uid}: x={node.x:.6f} m, y={node.y:.6f} m")
    
    print("\nApplied Loads (converted to SI for internal calculations):")
    for node in nodes:
        if node.loads:
            print(f"  Node {node.uid}:")
            for case, loads in node.loads.items():
                print(f"    {case}: Fx={loads[0]:.2f} N, Fy={loads[1]:.2f} N, Mz={loads[2]:.2f} N*m")
    
    print("\nNode Displacements:")
    for node in nodes:
        if hasattr(node, 'displacements') and load_combo.name in node.displacements:
            disp = node.displacements[load_combo.name]
            print(f"  Node {node.uid}: Ux={disp[0]:8.6f} in, Uy={disp[1]:8.6f} in, Rz={disp[2]:8.6f} rad")
    
    print("\nNode Reactions:")
    for node in nodes:
        if hasattr(node, 'reactions') and load_combo.name in node.reactions:
            reaction = node.reactions[load_combo.name]
            print(f"  Node {node.uid}: Rx={reaction[0]:8.2f} {IMPERIAL_UNITS['force']}, Ry={reaction[1]:8.2f} {IMPERIAL_UNITS['force']}, Mz={reaction[2]:8.2f} {IMPERIAL_UNITS['moment']}")
    
    return structure, load_combo

def format_load_list(loads, decimals=1):
    """Format load list with appropriate engineering precision"""
    return f"[{loads[0]:.{decimals}f}, {loads[1]:.{decimals}f}, {loads[2]:.{decimals}f}]"

if __name__ == "__main__":
    demo_excel_export()