# Example usage of the new export_results_to_excel method

import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyMAOS.nodes import R2Node
from pyMAOS.elements import R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo

# Create a simple 2-span beam example
def demo_excel_export():
    """
    Demonstrates the new export_results_to_excel method with a simple beam structure
    """
    
    # Create nodes
    N1 = R2Node(0, 0)      # Left support
    N2 = R2Node(10, 0)     # Middle support  
    N3 = R2Node(20, 0)     # Right support
    
    # Apply restraints
    N1.restrainAll()       # Fixed support
    N2.restrainTranslation()  # Pin support
    N3.restrainTranslation()  # Pin support
    
    # Node list
    nodes = [N1, N2, N3]
    
    # Apply loads
    loadcase = "D"
    N2.loads[loadcase] = [0, -50, 0]  # 50 unit downward load at middle node
    
    # Create material and section
    steel = Material(200e9, 7850, 200e9, 0.3)  # Steel material
    beam_section = Section(0.01, 8.33e-5)      # Typical beam section
    
    # Create frame members
    beam1 = R2Frame(N1, N2, steel, beam_section)
    beam2 = R2Frame(N2, N3, steel, beam_section)
    
    # Member list
    members = [beam1, beam2]
    
    # Add some member loads for demonstration
    beam1.add_distributed_load(-10, -10, 0, 10, case="D")  # Uniform load on first span
    beam2.add_point_load(-25, 5, case="D")                 # Point load on second span
    
    # Create structure
    structure = R2Struct.R2Structure(nodes, members)
    structure.set_node_uids()
    structure.set_member_uids()
    
    # Create load combination
    load_combo = LoadCombo("Dead Load", {"D": 1.0}, ["D"], False, "SLS")
    
    # Solve the structure
    print("Solving structure...")
    structure.solve_linear_static(load_combo, verbose=True)
    
    # Export results to Excel
    print("\nExporting results to Excel...")
    excel_file = structure.export_results_to_excel(
        load_combo, 
        output_file="beam_analysis_results.xlsx",
        include_visualization=True
    )
    
    print(f"Excel export completed: {excel_file}")
    
    # You can also export without visualization
    excel_file_no_viz = structure.export_results_to_excel(
        load_combo,
        output_file="beam_analysis_results_no_viz.xlsx", 
        include_visualization=False
    )
    
    print(f"Excel export (no visualization) completed: {excel_file_no_viz}")
    
    return structure, load_combo

if __name__ == "__main__":
    demo_excel_export()