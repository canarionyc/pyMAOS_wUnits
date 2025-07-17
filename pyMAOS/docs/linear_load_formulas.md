# Linear Load Formulas

## Overview

This document explains the mathematical formulas used in the `R2_Linear_Load` class for structural analysis of linearly varying distributed loads.

## Load Model

A linearly varying distributed load is defined by:
- Load intensities $w_1$ and $w_2$ at positions $a$ and $b$ respectively
- The load acts on a portion of the member from position $a$ to $b$
- Total member length is $L$

![Linear Load Diagram](https://via.placeholder.com/600x200?text=Linear+Load+Diagram)

## Simple Support Reactions

### Total Load (Area under load distribution)

The total load is calculated as the area under the trapezoidal load distribution:

$$W = \frac{1}{2} \cdot c \cdot (w_2 + w_1)$$

where $c = b - a$ is the length of the loaded region.

### Load Centroid

The position of the load centroid relative to point $a$ is:

$$\bar{c} = \frac{w_1 + 2w_2}{3(w_2 + w_1)} \cdot c$$

This formula locates the center of gravity of the trapezoidal load distribution.

### Simple Support Reactions

For a simply supported beam (before considering fixed-end conditions):

$$R_{j,y} = -W \cdot \frac{a + \bar{c}}{L}$$

$$R_{i,y} = -W - R_{j,y}$$

These formulas represent the vertical reactions at supports assuming the member is simply supported. The negative signs account for the sign convention where downward loads are negative.

## Fixed-End Forces

The fixed-end forces are calculated using integration constants ($c_1$ through $c_{12}$) that enforce boundary conditions:

$$M_{i,z} = -\frac{c_3 \cdot L^2 + 2c_6 \cdot L + 2c_9 + 4c_7}{L}$$

$$M_{j,z} = -\frac{2c_3 \cdot L^2 + 4c_6 \cdot L + 4c_9 + 2c_7}{L}$$

The final support reactions accounting for fixed-end moments:

$$R_{i,y} = R_{i,y} + \frac{M_{i,z}}{L} + \frac{M_{j,z}}{L}$$

$$R_{j,y} = R_{j,y} - \frac{M_{i,z}}{L} - \frac{M_{j,z}}{L}$$

## Piecewise Functions

The class defines piecewise polynomial functions for:

1. **Wy** - Load distribution function:
   - 0 in region [0, a]
   - Linear function in region [a, b]
   - 0 in region [b, L]

2. **Vy** - Shear force distribution (integral of Wy)

3. **Mz** - Bending moment distribution (integral of Vy)

4. **Sz** - Rotation/slope distribution (integral of Mz/EI)

5. **Dy** - Deflection distribution (integral of Sz)

Each function is defined with different polynomial coefficients in three regions:
- Before the load [0, a]
- Within the loaded region [a, b]
- After the load [b, L]

The continuity of these functions and their derivatives at the boundaries, along with the boundary conditions at the member ends, determine the integration constants.

## Numerical Example

Consider a beam with:
- Length L = 5m
- Linear load from a = 1m to b = 3m
- Load intensity w₁ = 2 kN/m at a
- Load intensity w₂ = 4 kN/m at b

Total load:
W = 0.5 × (3-1) × (2+4) = 6 kN

Load centroid:
c̄ = (2+2×4)/(3×(2+4)) × 2 = 10/18 × 2 = 1.111m from point a

Simple reactions:
Rⱼᵧ = -6 × (1+1.111)/5 = -2.533 kN
Rᵢᵧ = -6 - (-2.533) = -3.467 kN