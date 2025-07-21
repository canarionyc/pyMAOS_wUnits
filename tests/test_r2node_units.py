#!/usr/bin/env python3
"""
Test script to demonstrate R2Node coordinate parsing with units
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyMAOS.R2Node import R2Node

def test_r2node_unit_parsing():
    """Test R2Node creation with various coordinate formats"""
    
    print("Testing R2Node coordinate parsing with units...")
    print("=" * 60)
    
    # Test cases with different coordinate formats
    test_cases = [
        # (uid, x, y, description)
        (1, 0, 0, "Standard numeric coordinates (SI units)"),
        (2, "120in", "0in", "Imperial coordinates in inches"),
        (3, "10ft", "5ft", "Imperial coordinates in feet"),
        (4, "3.048m", "1.524m", "Metric coordinates in meters"),
        (5, "3048mm", "1524mm", "Metric coordinates in millimeters"),
        (6, 120.0, "10ft", "Mixed: float x, string y with units"),
        (7, "0", "0", "String coordinates without units"),
    ]
    
    nodes = []
    
    for uid, x, y, description in test_cases:
        try:
            print(f"\nTest {uid}: {description}")
            print(f"  Input: x={x}, y={y}")
            
            # Create the node
            node = R2Node(uid, x, y)
            nodes.append(node)
            
            print(f"  Result: x={node.x:.6f}m, y={node.y:.6f}m")
            
            # For imperial tests, show the conversion
            if isinstance(x, str) and ("in" in str(x) or "ft" in str(x)):
                print(f"  (Converted from imperial to SI units)")
            elif isinstance(x, str) and ("mm" in str(x)):
                print(f"  (Converted from mm to meters)")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Summary of created nodes:")
    print("=" * 60)
    
    for node in nodes:
        print(f"Node {node.uid}: ({node.x:.6f}m, {node.y:.6f}m)")
    
    print("\n" + "=" * 60)
    print("Detailed node information:")
    print("=" * 60)
    
    for node in nodes:
        print(f"\n{node}")
    
    # Test distance calculation between nodes
    if len(nodes) >= 2:
        print(f"\nDistance calculations:")
        print(f"Distance between Node 1 and Node 2: {nodes[0].distance(nodes[1]):.6f}m")
        if len(nodes) >= 3:
            print(f"Distance between Node 2 and Node 3: {nodes[1].distance(nodes[2]):.6f}m")

if __name__ == "__main__":
    test_r2node_unit_parsing()