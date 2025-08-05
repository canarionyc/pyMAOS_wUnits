# pyMAOS - Python Matrix Analysis of Structures

pyMAOS is a Python package for structural engineering analysis that provides powerful tools for analyzing and designing structural systems using matrix methods and finite element approaches.

## Key Features

### Consistent Unit Handling
- **Internal Unit System**: All calculations use consistent internal units (configurable)
- **Display Unit System**: Results can be displayed in any desired unit system (SI, Imperial, or custom)
- **Unit Conversion**: Seamless conversion between unit systems with Pint
- **Unit-Aware Calculations**: Mathematical operations preserve units throughout analysis

### Streamlined Data Workflows
- **Modern Input Options**: Import structural data from JSON or YAML files
- **Flexible Output Formats**: Export results to Excel or JSON for reporting and post-processing
- **Configuration Files**: Define unit systems, material properties, and section properties in external files

### Enhanced Visualization
- **3D Result Visualization**: View structural deformation and internal forces in three dimensions
- **Interactive Plots**: Manipulate and explore results visually
- **Customizable Graphics**: Tailor visualizations for reports and presentations

### Reactive Parametric Design (In Progress)
- **Automatic Recalculation**: Changes to input parameters trigger recomputation of results
- **Design Optimization**: Quickly evaluate design alternatives
- **Sensitivity Analysis**: Understand how changes affect structural performance

### Advanced Analysis Methods
- **Matrix Analysis**: Efficient structural analysis using direct stiffness method
- **Finite Element Analysis**: (In Progress) Resolve special cases requiring more detailed modeling
- **Custom Load Definitions**: Define arbitrary load distributions with piecewise polynomials

### Performance Optimized
- **Vectorized Operations**: Leverages NumPy for efficient numerical calculations
- **Scientific Computing**: Utilizes SciPy for specialized mathematical operations
- **Optimized Algorithms**: Fast solution of large structural systems

## Installation

```bash
pip install pyMAOS
```

## Quick Start

```python
import pyMAOS
from pyMAOS.pymaos_units import set_unit_system, IMPERIAL_UNITS

# Set preferred unit system
set_unit_system(IMPERIAL_UNITS, "imperial")

# Create a new structural model
model = pyMAOS.StructuralModel()

# Load model data from JSON
model.load_from_json("structure_definition.json")

# Analyze the structure
model.analyze()

# Export results to Excel
model.export_results("analysis_results.xlsx")

# Display deformation plot
model.plot_deformation(scale=10)
```

## Documentation

Comprehensive documentation is available at [https://pymaos.readthedocs.io/](https://pymaos.readthedocs.io/)

## The Unit System
The UnitManager singleton provides several advantages over using a global unit registry:

1. **Centralized Unit Handling**: All unit operations go through a single access point, ensuring consistency throughout the application.

2. **State Management**: Maintains the current unit system and preferences in one place, avoiding scattered global variables.

3. **Notification System**: Implements a registry mechanism to notify dependent modules when unit systems change:
   ```python
   def register_for_unit_updates(self, update_function):
       """Register a module to receive unit system updates"""
       self.registered_modules.append(update_function)
   ```

4. **Encapsulated Methods**: Provides specialized methods for common unit operations:

5. **Controlled Registry Access**: Prevents accidental registry configuration changes by exposing the registry through controlled interfaces.

6. **Simplified Testing**: Makes it easier to mock unit behavior for testing by targeting a single object.

7. **Eliminates Import Inconsistencies**: Prevents issues where different parts of code use different registry instances, which can lead to subtle unit conversion bugs.

8. **Cleaner API**: Provides a cleaner, more object-oriented approach compared to accessing global variables directly.

# Advantages of scipy's PPoly over numpy's Polynomial

SciPy's PPoly (Piecewise Polynomial) class offers several advantages over numpy's Polynomial:

1. **Piecewise representation**: PPoly can represent different polynomial segments over different intervals, while numpy's Polynomial represents a single polynomial over its entire domain.

2. **Memory efficiency**: For complex functions, using multiple lower-degree polynomials (as in PPoly) is more memory efficient than one high-degree polynomial.

3. **Numerical stability**: Lower-degree piecewise polynomials are more numerically stable than single high-degree polynomials.

4. **Efficient evaluation**: PPoly is optimized for fast evaluation of piecewise functions.

5. **Vectorized operations**: Both support vectorized operations, but PPoly is specifically optimized for piecewise evaluation.

6. **Local behavior modeling**: PPoly can model functions with different behaviors in different regions.

7. **Discontinuity handling**: Can represent functions with discontinuities between segments.

8. **Integration with splines**: PPoly is the underlying representation for many of scipy's spline interpolation methods.

In your specific code, PPoly is being used to handle polynomial segments of potentially different degrees across intervals, which is exactly what it's designed for.

The code history shows this evolution from global variables (causing inconsistencies) toward the more manageable singleton approach.

## Examples

The `examples/` directory contains several examples demonstrating the use of pyMAOS for different structural analysis applications:

- Simple beams and frames
- Trusses
- 3D structures
- Dynamic analysis
- Unit conversion examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

pyMAOS builds upon classical matrix structural analysis methods as described in:
- "Matrix Analysis of Structures" by Aslam Kassimali
- "Fundamentals of Structural Analysis" by Kenneth Leet, Chia-Ming Uang, and Joel Lanning