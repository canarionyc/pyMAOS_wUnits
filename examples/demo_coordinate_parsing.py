#!/usr/bin/env python3
"""
Simple demonstration of R2Node coordinate parsing with units
"""

# Example of creating nodes with different coordinate formats
from pyMAOS.R2Node import R2Node

def demo_coordinate_parsing():
    """Demonstrate the new coordinate parsing functionality"""
    
    print("R2Node Coordinate Parsing with Units - Demonstration")
    print("=" * 60)
    
    # Example 1: Traditional numeric coordinates (SI units assumed)
    node1 = R2Node(1, 0.0, 0.0)
    print(f"Node 1 (numeric): {node1.x:.6f}m, {node1.y:.6f}m")
    
    # Example 2: Imperial coordinates in inches
    node2 = R2Node(2, "120in", "0in")
    print(f"Node 2 (120in, 0in): {node2.x:.6f}m, {node2.y:.6f}m")
    
    # Example 3: Imperial coordinates in feet  
    node3 = R2Node(3, "10ft", "5ft")
    print(f"Node 3 (10ft, 5ft): {node3.x:.6f}m, {node3.y:.6f}m")
    
    # Example 4: Metric coordinates in millimeters
    node4 = R2Node(4, "3000mm", "1500mm")
    print(f"Node 4 (3000mm, 1500mm): {node4.x:.6f}m, {node4.y:.6f}m")
    
    # Example 5: Mixed formats
    node5 = R2Node(5, 1.5, "6ft")
    print(f"Node 5 (1.5, 6ft): {node5.x:.6f}m, {node5.y:.6f}m")
    
    print("\n" + "=" * 60)
    print("All coordinates are internally stored in SI units (meters)")
    print("This ensures consistent calculations throughout the analysis")

if __name__ == "__main__":
    demo_coordinate_parsing()