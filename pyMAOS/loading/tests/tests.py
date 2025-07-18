"""
Tests for the loading module, particularly to verify R2_Linear_Load
behaves correctly for a uniform load case.
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_uniform_load():
    """
    Test that R2_Linear_Load produces the expected fixed end forces 
    for a uniform load spanning the entire element.
    """
    import numpy as np
    from pyMAOS.loading.distributed_loads import R2_Linear_Load
    
    # Create a simple member with length=10
    L = 10.0
    w = 5.0  # uniform load
    
    # Create dummy member
    class DummyMember:
        def __init__(self):
            self.length = L
            self.material = type('obj', (object,), {'E': 200e9})
            self.section = type('obj', (object,), {'Ixx': 1e-4})
    
    member = DummyMember()
    
    # Create the uniform load (w1=w2, spans entire member)
    load = R2_Linear_Load(w1=w, w2=w, a=0, b=L, member=member)
    
    # Calculate FEF
    fef = load.FEF()
    
    # Expected values for uniform load of intensity w on span L
    Miz_expected = -w * L**2 / 12
    Mjz_expected = w * L**2 / 12
    Riy_expected = w * L / 2
    Rjy_expected = w * L / 2
    
    print(f"Uniform load w={w} on span L={L}:")
    print(f"Miz: expected={Miz_expected:.4f}, actual={fef[2]:.4f}")
    print(f"Mjz: expected={Mjz_expected:.4f}, actual={fef[5]:.4f}")
    print(f"Riy: expected={Riy_expected:.4f}, actual={fef[1]:.4f}")
    print(f"Rjy: expected={Rjy_expected:.4f}, actual={fef[4]:.4f}")
    
    # Check if values match within tolerance
    tol = 1e-6
    matches = (
        abs(fef[2] - Miz_expected) < tol and
        abs(fef[5] - Mjz_expected) < tol and
        abs(fef[1] - Riy_expected) < tol and
        abs(fef[4] - Rjy_expected) < tol
    )
    
    print(f"Values match expected: {matches}")
    return matches

if __name__ == "__main__":
    test_uniform_load()