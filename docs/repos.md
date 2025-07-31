There are several online repositories that provide structural models suitable for analysis with a Python-based FEM project like yours:

1. **NEES Repository** - Contains experimental data and models from earthquake engineering research: https://www.designsafe-ci.org/data/

2. **OpenSees Model Database** - Open System for Earthquake Engineering Simulation provides structural models with focus on seismic applications: https://opensees.berkeley.edu/

3. **SDC Verifier Examples** - Contains benchmark examples for structural analysis verification: https://sdcverifier.com/examples/

4. **StructX** - Collection of structural engineering examples: https://structx.com/

5. **NAFEMS Benchmark Models** - Standard benchmark problems for FEA validation: https://www.nafems.org/publications/resource_center/

You could adapt models from these sources by:

```python
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import json

def load_model_from_json(filepath):
    """Load structural model definition from JSON file."""
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # Extract nodes, elements, materials
    nodes = np.array(model_data['nodes'])
    elements = np.array(model_data['elements'])
    
    print(f"DEBUG: Loaded model with {len(nodes)} nodes and {len(elements)} elements")
    return nodes, elements, model_data.get('materials', {})

# Example usage with your Bernstein shape functions
def analyze_model(nodes, elements, materials, order=3):
    results = []
    
    for element in elements:
        # Get element nodes
        node_indices = element['connectivity']
        element_nodes = nodes[node_indices]
        
        # Generate shape functions for this element
        local_coords = np.linspace(0, 1, 10)  # Local coordinates for integration
        shape_functions = bernstein_shape_functions(order, local_coords)
        
        # Calculate element properties (simplified)
        # In a real implementation, you would compute stiffness matrix, etc.
        results.append({
            'element_id': element['id'],
            'shape_functions': shape_functions,
            'nodes': element_nodes
        })
        
    return results
```

Most repositories provide models in standard formats like ABAQUS, ANSYS, or NASTRAN input files that would need parsing before use with your shape functions.