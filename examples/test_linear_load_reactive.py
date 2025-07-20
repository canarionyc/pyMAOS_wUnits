import os
import sys
import numpy as np
from rx import operators as ops

# Ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyMAOS.linear_load_reactive import LinearLoadReactive

def test_basic_calculation():
    """Test basic calculation of linear load using reactive approach"""
    print("\n=== Testing Basic Linear Load Calculation ===")
    
    # Create reactive load processor
    load = LinearLoadReactive()
    
    # Define test parameters
    w1 = -10.0  # N/m (negative for downward load)
    w2 = -20.0  # N/m
    a = 1.0     # m
    b = 4.0     # m
    L = 5.0     # m
    
    # Set parameters in reactive system
    load.set_parameters(w1, w2, a, b, L)
    
    # Expected results based on formulas
    c = b - a
    expected_total = 0.5 * c * (w1 + w2)
    expected_centroid = c * (w1 + 2*w2) / (3*(w1 + w2))
    expected_rj = -expected_total * (a + expected_centroid) / L
    expected_ri = -expected_total - expected_rj
    
    # Get actual results using rx operators
    actual_total = None
    actual_centroid = None
    actual_ri = None
    actual_rj = None
    
    # Subscribe to observe results
    load.total_load.pipe(ops.first()).subscribe(lambda x: globals().update(actual_total=x))
    load.load_centroid.pipe(ops.first()).subscribe(lambda x: globals().update(actual_centroid=x))
    load.reactions.pipe(ops.first()).subscribe(lambda x: globals().update(actual_ri=x[0], actual_rj=x[1]))
    
    # Wait briefly to ensure values are updated
    import time
    time.sleep(0.1)
    
    # Print comparison
    print(f"Input: w1={w1}, w2={w2}, a={a}, b={b}, L={L}")
    print(f"Total load:   Expected={expected_total:.4f}, Actual={actual_total:.4f}")
    print(f"Load centroid: Expected={expected_centroid:.4f}, Actual={actual_centroid:.4f}")
    print(f"Reaction Ri:   Expected={expected_ri:.4f}, Actual={actual_ri:.4f}")
    print(f"Reaction Rj:   Expected={expected_rj:.4f}, Actual={actual_rj:.4f}")
    
    # Check if calculations are correct (within tolerance)
    tol = 1e-6
    is_correct = (abs(actual_total - expected_total) < tol and
                 abs(actual_centroid - expected_centroid) < tol and
                 abs(actual_ri - expected_ri) < tol and
                 abs(actual_rj - expected_rj) < tol)
    
    print(f"All calculations correct: {is_correct}")
    return is_correct

def test_reactive_updates():
    """Test reactive updates when input parameters change"""
    print("\n=== Testing Reactive Updates ===")
    
    # Create reactive load processor
    load = LinearLoadReactive()
    
    # Initial parameters
    w1 = -10.0
    w2 = -10.0
    a = 1.0
    b = 4.0
    L = 5.0
    
    # Collections to store results
    results = []
    
    # Subscribe to combined results
    rx.combine_latest(
        load.total_load,
        load.load_centroid,
        load.reactions
    ).subscribe(lambda values: results.append({
        'total': values[0],
        'centroid': values[1],
        'ri': values[2][0],
        'rj': values[2][1]
    }))
    
    # Set initial parameters
    load.set_parameters(w1, w2, a, b, L)
    
    # Change w2 incrementally and observe reactive updates
    print("Changing w2 and observing updates:")
    for delta in range(5):
        new_w2 = w2 - 5.0 * delta
        print(f"\nChanging w2 to {new_w2}")
        load.w2_subject.on_next(new_w2)
        
        # Wait briefly for reactive processing
        import time
        time.sleep(0.1)
        
        # Print the latest result
        if results:
            latest = results[-1]
            print(f"  Total load: {latest['total']:.4f}")
            print(f"  Centroid: {latest['centroid']:.4f}")
            print(f"  Reactions: Ri={latest['ri']:.4f}, Rj={latest['rj']:.4f}")
    
    print(f"\nTotal updates received: {len(results)}")
    return len(results) > 1

def integrate_with_frame_loader():
    """Test integration with frame loader using a mock element"""
    print("\n=== Testing Integration with Frame Loader ===")
    
    # Mock element class that records loads
    class MockElement:
        def __init__(self, uid, length):
            self.uid = uid
            self.length = length
            self.loads = []
        
        def add_distributed_load(self, w1, w2, a, b, load_case, direction="Y"):
            self.loads.append({
                'type': 'distributed',
                'w1': w1, 
                'w2': w2,
                'a': a,
                'b': b,
                'case': load_case,
                'direction': direction
            })
            print(f"Added distributed load to element {self.uid}: w1={w1}, w2={w2}")
            return True
    
    # Create mock element and load data
    element = MockElement(1, 5.0)
    member_load = {
        "member_uid": 1,
        "load_type": 3,
        "wi": -10.0,
        "wj": -20.0,
        "a": 1.0,
        "b": 4.0,
        "case": "D",
        "direction": "Y"
    }
    
    # Process load using your reactive function
    from load_frame_from_json import load_linear_load_reactively
    
    try:
        load_linear_load_reactively(element, member_load)
        print("Successfully processed load reactively")
        print(f"Element has {len(element.loads)} loads")
        
        if element.loads:
            print(f"Load details: {element.loads[0]}")
        return True
    except Exception as e:
        print(f"Error in reactive load processing: {e}")
        return False

if __name__ == "__main__":
    import rx
    
    # Run tests
    print("===== LINEAR LOAD REACTIVE TESTS =====")
    basic_ok = test_basic_calculation()
    reactive_ok = test_reactive_updates()
    
    # Uncomment to test integration (requires proper imports)
    # integration_ok = integrate_with_frame_loader()
    
    print("\n===== TEST SUMMARY =====")
    print(f"Basic calculation: {'PASS' if basic_ok else 'FAIL'}")
    print(f"Reactive updates: {'PASS' if reactive_ok else 'FAIL'}")
    # print(f"Integration test: {'PASS' if integration_ok else 'FAIL'}")
    
    # Optionally plot results with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        # Create reactive load processor
        load = LinearLoadReactive()
        
        # Parameters
        w1 = -10.0
        a = 1.0
        b = 4.0
        L = 5.0
        
        # Arrays to collect data
        w2_values = np.linspace(-10, -30, 20)
        total_loads = []
        centroids = []
        ri_values = []
        rj_values = []
        
        # Subscribe to collect results
        load.total_load.subscribe(lambda x: total_loads.append(x))
        load.load_centroid.subscribe(lambda x: centroids.append(x))
        load.reactions.subscribe(lambda x: (ri_values.append(x[0]), rj_values.append(x[1])))
        
        # Run calculations for different w2 values
        for w2 in w2_values:
            load.set_parameters(w1, w2, a, b, L)
            
        # Plot results
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(w2_values, total_loads, 'b-')
        plt.title('Total Load vs w2')
        plt.xlabel('w2 (N/m)')
        plt.ylabel('Total Load (N)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(w2_values, centroids, 'g-')
        plt.title('Centroid Position vs w2')
        plt.xlabel('w2 (N/m)')
        plt.ylabel('Position from a (m)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(w2_values, ri_values, 'r-')
        plt.title('Reaction Ri vs w2')
        plt.xlabel('w2 (N/m)')
        plt.ylabel('Ri (N)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(w2_values, rj_values, 'm-')
        plt.title('Reaction Rj vs w2')
        plt.xlabel('w2 (N/m)')
        plt.ylabel('Rj (N)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('linear_load_reactive_test.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
