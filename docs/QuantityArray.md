# QuantityArray Advantages and Limitations

The `QuantityArray` class in your code provides several advantages when working with unit-aware quantities:

## Advantages
1. **NumPy Integration**: It subclasses `np.ndarray`, maintaining compatibility with NumPy's ecosystem
2. **Automatic Magnitude Extraction**: Extracts numeric values from Pint quantities during assignment
3. **Simplified Operations**: Allows using NumPy functions that don't natively support Pint objects
4. **Custom Display**: With your `format_object_array` function, it provides better string representation for quantities
5. **Boundary Handling**: Good for cases where units matter at interfaces but not during computation

## Limitations with Mixed Quantities

QuantityArray **doesn't support mixed quantities** in its basic form because:

1. It extracts only magnitudes, losing unit information
2. No dimensional checking between operations
3. No automatic unit conversion

When you encountered the "object arrays not supported" error, it was because you were trying to use arrays containing full `pint.Quantity` objects rather than just their magnitudes.

## Alternative Approach

Your solution correctly handles quantities by:

```python
# Extract magnitudes for calculation
Kff_magnitudes = np.array([[k.magnitude if hasattr(k, 'magnitude') else float(k) for k in row]
                         for row in self.Kff], dtype=np.float64)

# Solve using numeric values
U_magnitudes = sla.solve(Kff_magnitudes, FGf_magnitudes - PFf_magnitudes)

# Reattach units afterward
U = np.array([unit_manager.ureg.Quantity(mag, conj_unit) for mag, conj_unit in zip(U_magnitudes, conjugate_units)], 
             dtype=object)
```

This pattern (extract → compute → reattach) is the most reliable way to handle mixed quantities in scientific computing.