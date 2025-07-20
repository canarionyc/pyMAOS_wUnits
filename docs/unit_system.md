# Unit System in pyMAOS

pyMAOS uses a consistent unit management system throughout the package to ensure accurate calculations and convenient display of results.

## Supported Unit Systems

- **SI Units**: Newtons, meters, Pascals, etc.
- **Imperial Units**: Pounds-force, feet/inches, psi, etc.
- **Metric kN**: Kilonewtons, meters, kN/m², etc.

## How Units Work in pyMAOS

1. **Internal Calculations**: All internal calculations are performed using SI units for maximum precision and consistency.

2. **Display Units**: Results are displayed using the currently selected unit system.

3. **Input Parsing**: Input values can include unit specifications (e.g., "10 kN", "20 ft").

## Specifying Units in Input

You can specify units in your JSON input files:
```json
{ "units": { "force": "kN", "length": "m", "pressure": "kPa" }, "nodes": [...] }
```

Or use command-line arguments:
```code
python load_frame_from_json.py input.json --units imperial
```

## Supported Dimensions and Units

| Dimension | SI Unit | Imperial Unit | Metric kN Unit |
|-----------|---------|--------------|----------------|
| Force | N | lbf | kN |
| Length | m | ft | m |
| Pressure | Pa | psi | kPa |
| Moment | N·m | lbf·ft | kN·m |
| Area | m² | in² | m² |
| Moment of inertia | m⁴ | in⁴ | m⁴ |
| Distributed load | N/m | lbf/ft | kN/m |

