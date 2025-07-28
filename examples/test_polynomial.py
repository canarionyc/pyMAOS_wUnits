# Test file: test_polynomial.py
import pytest
from pyMAOS.loading.piecewisePolinomial import PiecewisePolynomial

def test_combine():
    # Create two PiecewisePolynomial objects
    poly1 = PiecewisePolynomial()
    poly1.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    poly2 = PiecewisePolynomial()
    poly2.add_segment(1, 2, [3, 4])  # Segment: 3 + 4*x

    # Combine with load factors
    combined = poly1.combine(poly2, LF=2, LFother=3)

    # Validate the combined polynomial
    assert combined.evaluate(0.5) == pytest.approx(2 * (1  + 2* 0.5))
    assert combined.evaluate(1.5) == pytest.approx(3 * (3 + 4 * 1.5))

def test_combine_invalid():
    poly1 = PiecewisePolynomial()
    poly2 = "InvalidType"  # Not a PiecewisePolynomial

    with pytest.raises(TypeError):
        poly1.combine(poly2, LF=1, LFother=1)

def test_evaluate():
    poly = PiecewisePolynomial()
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Evaluate at a point within the segment
    assert poly.evaluate(0.5) == pytest.approx(1 + 2 * 0.5)

    # Evaluate at a point outside the segment (should raise an error)
    with pytest.raises(ValueError):
        poly.evaluate(1.5)

def test_add_segment():
    poly = PiecewisePolynomial()

    # Add a segment and check if it exists
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x
    assert len(poly.segments) == 1
    assert poly.segments[0].start == 0
    assert poly.segments[0].end == 1
    assert poly.segments[0].coefficients == [1, 2]

    # Add another segment
    poly.add_segment(1, 2, [3, 4])  # Segment: 3 + 4*x
    assert len(poly.segments) == 2
    assert poly.segments[1].start == 1
    assert poly.segments[1].end == 2
    assert poly.segments[1].coefficients == [3, 4]

def test_add_segment_invalid():
    poly = PiecewisePolynomial()

    # Attempt to add a segment with invalid coefficients
    with pytest.raises(ValueError):
        poly.add_segment(0, 1, [1, "invalid"])  # Coefficients must be numeric

def test_empty_polynomial():
    poly = PiecewisePolynomial()

    # Evaluate an empty polynomial (should raise an error)
    with pytest.raises(ValueError):
        poly.evaluate(0)  # No segments to evaluate

def test_segment_overlap():
    poly = PiecewisePolynomial()

    # Add a segment
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Attempt to add an overlapping segment
    with pytest.raises(ValueError):
        poly.add_segment(0.5, 1.5, [3, 4])  # Overlaps with existing segment

def test_segment_non_continuous():
    poly = PiecewisePolynomial()

    # Add a segment
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Attempt to add a non-continuous segment
    with pytest.raises(ValueError):
        poly.add_segment(1.5, 2.5, [3, 4])  # Does not connect to previous segment

def test_segment_order():
    poly = PiecewisePolynomial()

    # Add segments in order
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x
    poly.add_segment(1, 2, [3, 4])  # Segment: 3 + 4*x

    # Check if segments are ordered correctly
    assert poly.segments[0].start == 0
    assert poly.segments[1].start == 1

    # Attempt to add a segment out of order
    with pytest.raises(ValueError):
        poly.add_segment(0.5, 1.5, [5, 6])  # Should not be allowed

def test_segment_boundary():
    poly = PiecewisePolynomial()

    # Add a segment at the boundary
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Check if the segment exists at the boundary
    assert len(poly.segments) == 1
    assert poly.segments[0].start == 0
    assert poly.segments[0].end == 1

    # Attempt to add a segment that overlaps the boundary
    with pytest.raises(ValueError):
        poly.add_segment(0.5, 1.5, [3, 4])  # Should not overlap with existing segment

def test_segment_evaluate():
    poly = PiecewisePolynomial()

    # Add a segment
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Evaluate at the start of the segment
    assert poly.evaluate(0) == pytest.approx(1)  # 1 + 2*0

    # Evaluate at the end of the segment
    assert poly.evaluate(1) == pytest.approx(3)  # 1 + 2*1

    # Evaluate in the middle of the segment
    assert poly.evaluate(0.5) == pytest.approx(2)  # 1 + 2*0.5

def test_segment_evaluate_outside():
    poly = PiecewisePolynomial()

    # Add a segment
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Attempt to evaluate outside the segment range
    with pytest.raises(ValueError):
        poly.evaluate(-0.5)  # Outside the segment range
    with pytest.raises(ValueError):
        poly.evaluate(1.5)  # Outside the segment range

def test_segment_evaluate_boundary():
    poly = PiecewisePolynomial()

    # Add a segment
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x

    # Evaluate at the boundary points
    assert poly.evaluate(0) == pytest.approx(1)  # At start
    assert poly.evaluate(1) == pytest.approx(3)  # At end

    # Attempt to evaluate just outside the boundary
    with pytest.raises(ValueError):
        poly.evaluate(-0.1)  # Just outside the start
    with pytest.raises(ValueError):
        poly.evaluate(1.1)  # Just outside the end

def test_segment_evaluate_multiple_segments():
    poly = PiecewisePolynomial()

    # Add multiple segments
    poly.add_segment(0, 1, [1, 2])  # Segment: 1 + 2*x
    poly.add_segment(1, 2, [3, 4])  # Segment: 3 + 4*x

    # Evaluate at points in each segment
    assert poly.evaluate(0.5) == pytest.approx(2)  # In first segment
    assert poly.evaluate(1.5) == pytest.approx(7)  # In second segment

    # Attempt to evaluate at the boundaries of each segment
    assert poly.evaluate(0) == pytest.approx(1)  # Start of first segment
    assert poly.evaluate(1) == pytest.approx(3)  # End of first segment and start of second
    assert poly.evaluate(2) == pytest.approx(11)  # End of second segment

    # Attempt to evaluate outside the combined range
    with pytest.raises(ValueError):
        poly.evaluate(-0.5)  # Outside both segments
    with pytest.raises(ValueError):
        poly.evaluate(2.5)  # Outside both segments

if __name__ == "__main__":
    pytest.main([__file__])

    # Example usage of UnitAwarePolynomial with units
    from pyMAOS.unit_aware import ureg
    from pyMAOS.loading.UnitAwarePolynomial import UnitAwarePolynomial
    # Create a polynomial with units
    coeffs = [2 * ureg.newton, 3 * ureg.newton / ureg.meter]
    poly = UnitAwarePolynomial(coeffs, y_units=ureg.newton, x_units=ureg.meter)

    # Evaluate at multiple points
    x_values = [0, 1, 2, 3] * ureg.meter
    y_values = poly(x_values)  # Returns array of values with newton units
    print(y_values)  # [2 newton, 5 newton, 8 newton, 11 newton]