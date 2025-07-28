# Linear Load Formulas

## Overview

This document explains the mathematical formulas used in the `LinearLoadXY` class for structural analysis of linearly varying distributed loads.

## Load Model

A linearly varying distributed load is defined by:

-   Load intensities $w_1$ and $w_2$ at positions $a$ and $b$ respectively
-   The load acts on a portion of the member from position $a$ to $b$
-   Total member length is $L$

## Simple Support Reactions

### Total Load (Area under load distribution)

The total load is calculated as the area under the trapezoidal load distribution:

$$W = \frac{1}{2} c (w_2 + w_1)$$

where $c = b - a$ is the length of the loaded region.

### Load Centroid

The position of the load centroid relative to point $a$ is:

$$\bar{c} = \frac{w_1 + 2w_2}{3(w_2 + w_1)} c$$

This formula locates the center of gravity of the trapezoidal load distribution.

### Simple Support Reactions

For a simply supported beam (before considering fixed-end conditions):

$$R_{j,y} = -W \frac{a + \bar{c}}{L}$$

$$R_{i,y} = -W - R_{j,y}$$

These formulas represent the vertical reactions at supports assuming the member is simply supported. The negative signs account for the sign convention where downward loads are negative.

## Integration Constants

The integration constants ($c_1$ through $c_{12}$) are derived by enforcing boundary conditions and continuity requirements across the three regions of the beam. These constants are essential for defining the piecewise polynomial functions for shear, moment, slope, and deflection.

### Shear Force Constants ($c_1$, $c_2$, $c_3$)

These constants define the shear force in the three regions:

-   $c_1$: Constant shear in region [0, a]
-   $c_2$: Base constant for shear in region [a, b]
-   $c_3$: Constant shear in region [b, L]

$$c_1 = \frac{(2b^2 + (-a-3L)b - a^2 + 3La)w_2 + (b^2 + (a-3L)b - 2a^2 + 3La)w_1}{6L}$$

$$c_2 = \frac{(2b^3 + (-3a-3L)b^2 + 6Lab + a^3)w_2 + (b^3 - 3Lb^2 - 3a^2b + 2a^3)w_1}{6L(b-a)}$$

$$c_3 = \frac{(2b^2 - ab - a^2)w_2 + (b^2 + ab - 2a^2)w_1}{6L}$$

### Bending Moment Constants ($c_4$, $c_5$, $c_6$)

These constants define the bending moment functions:

-   $c_4 = 0$: Zero moment at x=0 (for fixed-end condition)
-   $c_5$: Base constant for moment in region [a, b]
-   $c_6$: Base constant for moment in region [b, L]

$$c_4 = 0$$

$$c_5 = \frac{-1(a^3w_2 + (2a^3 - 3a^2b)w_1)}{6(b-a)}$$

$$c_6 = \frac{-1((2b^2 - ab - a^2)w_2 + (b^2 + ab - 2a^2)w_1)}{6}$$

### Slope/Rotation Constants ($c_7$, $c_8$, $c_9$)

These constants define the slope functions after dividing by $E I$ (flexural rigidity). They are derived from enforcing continuity conditions and boundary conditions for beam rotation.

#### $c_7$: Constant for Region [0, a]

$c_7$ influences the rotation in the first region of the beam and helps satisfy the boundary conditions at $x=0$.

$$c_7 = \frac{1}{360L} \Bigg[ w_2 \Big( 12b^4 + (-3a-45L)b^3 + (-3a^2+15La+40L^2)b^2 + (-3a^3+15La^2-20L^2a)b - 3a^4 + 15La^3 - 20L^2a^2 \Big)$$

$$+ w_1 \Big( 3b^4 + (3a-15L)b^3 + (3a^2-15La+20L^2)b^2 + (3a^3-15La^2+20L^2a)b - 12a^4 + 45La^3 - 40L^2a^2 \Big) \Bigg]$$

This complex polynomial ensures proper rotation behavior near the support at $x=0$ and continuity with the middle region.

#### $c_8$: Constant for Region [a, b]

$c_8$ is crucial for ensuring continuity of rotation at $x=a$ (where the load begins):

$$c_8 = \frac{1}{360L(b-a)} \Bigg[ w_2 \Big( 12b^5 + (-15a-45L)b^4 + (60La+40L^2)b^3 - 60L^2ab^2 + 3a^5 + 20L^2a^3 \Big)$$

$$+ w_1 \Big( 3b^5 - 15Lb^4 + 20L^2b^3 + (-15a^4-60L^2a^2)b + 12a^5 + 40L^2a^3 \Big) \Bigg]$$

This constant handles the transition between the unloaded and loaded portions of the beam, ensuring smooth rotation behavior.

#### $c_9$: Constant for Region [b, L]

$c_9$ ensures continuity of rotation at $x=b$ (where the load ends) and proper behavior toward the right support:

$$c_9 = \frac{1}{360L} \Bigg[ w_2 \Big( 12b^4 - 3ab^3 + (40L^2-3a^2)b^2 + (-3a^3-20L^2a)b - 3a^4 - 20L^2a^2 \Big)$$

$$+ w_1 \Big( 3b^4 + 3ab^3 + (3a^2+20L^2)b^2 + (3a^3+20L^2a)b - 12a^4 - 40L^2a^2 \Big) \Bigg]$$

This constant ensures proper rotation behavior in the region after the load and approaching the right support.

### Physical Significance

When divided by $E I$:

1. These constants define the shape of the rotation function across the beam.
2. They ensure continuity of rotation across region boundaries.
3. They maintain compatibility with the deflection and moment distributions.
4. They account for the fixed-end boundary conditions (zero rotation at supports for fixed-end beams).

The seemingly complex expressions result from solving the differential equation of the deflection curve while enforcing appropriate boundary conditions.

### Deflection Constants ($c_{10}$, $c_{11}$, $c_{12}$)

These constants define the deflection functions after dividing by $E I$:

-   $c_{10} = 0$: Zero deflection at x=0 (for fixed-end condition)
-   $c_{11}$: Base constant for deflection in region [a, b]
-   $c_{12}$: Base constant for deflection in region [b, L]

$$c_{10} = 0$$

$$c_{11} = \frac{-1(a^5w_2 + (4a^5 - 5a^4b)w_1)}{120(b-a)}$$

$$c_{12} = \frac{-1((4b^4 - ab^3 - a^2b^2 - a^3b - a^4)w_2 + (b^4 + ab^3 + a^2b^2 + a^3b - 4a^4)w_1)}{120}$$

## Fixed-End Forces

The fixed-end forces are calculated using these integration constants:

$$M_{i,z} = -\frac{c_3 L^2 + 2c_6 L + 2c_9 + 4c_7}{L}$$

$$M_{j,z} = -\frac{2c_3 L^2 + 4c_6 L + 4c_9 + 2c_7}{L}$$

The final support reactions accounting for fixed-end moments:

$$R_{i,y} = R_{i,y} + \frac{M_{i,z}}{L} + \frac{M_{j,z}}{L}$$

$$R_{j,y} = R_{j,y} - \frac{M_{i,z}}{L} - \frac{M_{j,z}}{L}$$

## Piecewise Functions

The class defines piecewise polynomial functions for:

1.  **Wy** - Load distribution function:

    -   0 in region [0, a]
    -   Linear function in region [a, b]
    -   0 in region [b, L]

2.  **Vy** - Shear force distribution (integral of Wy)

3.  **Mz** - Bending moment distribution (integral of Vy)

4.  **Sz** - Rotation/slope distribution (integral of Mz/EI)

5.  **Dy** - Deflection distribution (integral of Sz)

Each function is defined with different polynomial coefficients in three regions:

-   Before the load [0, a]
-   Within the loaded region [a, b]
-   After the load [b, L]

The continuity of these functions and their derivatives at the boundaries, along with the boundary conditions at the member ends, determine the integration constants.

## Numerical Example

Consider a beam with:

-   Length L = 5m
-   Linear load from a = 1m to b = 3m
-   Load intensity w₁ = 2 kN/m at a
-   Load intensity w₂ = 4 kN/m at b

Total load: W = 0.5 × (3-1) × (2+4) = 6 kN

Load centroid: c̄ = (2+2×4)/(3×(2+4)) × 2 = 10/18 × 2 = 1.111m from point a

Simple reactions: Rⱼᵧ = -6 × (1+1.111)/5 = -2.533 kN Rᵢᵧ = -6 - (-2.533) = -3.467 kN
