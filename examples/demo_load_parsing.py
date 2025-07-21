#!/usr/bin/env python3
"""
Simple demonstration of R2Node load parsing with units
"""

from pyMAOS.R2Node import R2Node

def demo_load_parsing():
    """Demonstrate the new load parsing functionality"""
    
    print("R2Node Load Parsing with Units - Demonstration")
    print("=" * 60)
    
    # Create a node with imperial coordinates
    node = R2Node(1, "10ft", "0ft")
    print(f"Node coordinates: ({node.x:.6f}m, {node.y:.6f}m) [converted from 10ft, 0ft]")
    
    print("\nLoad Assignment Examples:")
    print("-" * 30)
    
    # Example 1: Direct load assignment with imperial units
    node.loads["Dead"] = [0, "-50kip", 0]
    print(f"Dead Load: [0, '-50kip', 0] → SI: {node.loads['Dead']}")
    
    # Example 2: Using add_nodal_load method
    node.add_nodal_load("25kip", "-10kip", "100kip*ft", "Live")
    print(f"Live Load: add_nodal_load('25kip', '-10kip', '100kip*ft') → SI: {node.loads['Live']}")
    
    # Example 3: Mixed units and formats
    node.loads["Wind"] = ["15kN", "-25klbf", "50kN*m"]
    print(f"Wind Load: ['15kN', '-25klbf', '50kN*m'] → SI: {node.loads['Wind']}")
    
    print("\nAll loads in SI units (N, N*m):")
    print("-" * 40)
    node.display_loads()
    
    print("\nUnit Conversion Reference:")
    print("-" * 30)
    print("1 kip = 4,448.22 N")
    print("1 kip*ft = 1,355.82 N*m")
    print("1 klbf = 1,000 lbf = 4,448.22 N")
    print("1 kN = 1,000 N")

if __name__ == "__main__":
    demo_load_parsing()