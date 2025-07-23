#!/usr/bin/env python3
"""
Test script to demonstrate R2Node load parsing with units
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyMAOS.node2d import R2Node

def test_r2node_load_parsing():
    """Test R2Node load assignment with various unit formats"""
    
    print("Testing R2Node load parsing with units...")
    print("=" * 60)
    
    # Create a test node
    node = R2Node(1, 0, 0)
    
    print("Test 1: Direct load assignment with string units")
    print("-" * 50)
    
    # Test case 1: Load assignment with imperial units
    node.loads["D1"] = [0, "-50kip", 0]  # 50 kips downward
    print(f"Input: [0, '-50kip', 0]")
    print(f"Stored (SI): {node.loads['D1']} [N, N, N*m]")
    
    # Test case 2: Load assignment with metric units
    node.loads["D2"] = ["100kN", "-200kN", "50kN*m"]
    print(f"\nInput: ['100kN', '-200kN', '50kN*m']")
    print(f"Stored (SI): {node.loads['D2']} [N, N, N*m]")
    
    # Test case 3: Mixed numeric and string units
    node.loads["D3"] = [1000, "-25kip", "100klbf*ft"]
    print(f"\nInput: [1000, '-25kip', '100klbf*ft']")
    print(f"Stored (SI): {node.loads['D3']} [N, N, N*m]")
    
    print("\n" + "=" * 60)
    print("Test 2: Using add_nodal_load method with string units")
    print("-" * 50)
    
    # Test case 4: Using add_nodal_load method
    node.add_nodal_load("75kip", "-125kip", "200kip*ft", "L1")
    print(f"Input: add_nodal_load('75kip', '-125kip', '200kip*ft', 'L1')")
    print(f"Stored (SI): {node.loads['L1']} [N, N, N*m]")
    
    # Test case 5: Accumulating loads
    node.add_nodal_load("25kip", "0kip", "50kip*ft", "L1")  # Add to existing L1
    print(f"\nAdding: add_nodal_load('25kip', '0kip', '50kip*ft', 'L1')")
    print(f"Total (SI): {node.loads['L1']} [N, N, N*m]")
    
    print("\n" + "=" * 60)
    print("Test 3: Various unit formats")
    print("-" * 50)
    
    test_cases = [
        (["10lbf", "-20lbf", "5lbf*ft"], "Imperial: lbf, lbf*ft"),
        (["50klbf", "-100klbf", "200klbf*in"], "Imperial: klbf, klbf*in"),
        (["1000N", "-2000N", "500N*m"], "SI: N, N*m"),
        (["15kN", "-30kN", "75kN*m"], "Metric: kN, kN*m"),
        (["0", "0", "0"], "String zeros"),
        ([0, 0, 0], "Numeric zeros"),
    ]
    
    for i, (load_values, description) in enumerate(test_cases, 1):
        loadcase = f"T{i}"
        node.loads[loadcase] = load_values
        print(f"{description}:")
        print(f"  Input: {load_values}")
        print(f"  SI: {node.loads[loadcase]} [N, N, N*m]")
        print()
    
    print("=" * 60)
    print("Display all loads using the display_loads method:")
    print("=" * 60)
    node.display_loads()
    
    print("\n" + "=" * 60)
    print("Conversion verification:")
    print("=" * 60)
    print("50 kip = 50 � 4448.22 = 222,411 N")
    print("200 kip*ft = 200 � 4448.22 � 0.3048 = 271,140 N*m")
    print("100 kN = 100,000 N")
    print("50 kN*m = 50,000 N*m")

if __name__ == "__main__":
    test_r2node_load_parsing()