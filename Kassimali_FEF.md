# Beam Fixed-End Force Formulas

## Introduction

Fixed-end forces (FEF) are the forces and moments that develop at the ends of a beam when both ends are fixed against rotation and translation, and the beam is subjected to external loads. These forces are fundamental components in matrix structural analysis using the stiffness method.

The sign convention used in these formulas follows Kassimali's "Matrix Analysis of Structures":
- Positive moments cause compression in the top fibers of the beam
- Positive forces act upward

## Common Loading Cases

### 1. Concentrated Load (Point Load)

A point load P applied at a distance a from end A and b from end B (L = a + b).

```
    A                     B
    |------a----->P<---b--|
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = -\frac{Pb^2a}{L^2} $$
$$ M_{FB} = \frac{Pa^2b}{L^2} $$

#### Fixed-End Shears:
$$ V_{FA} = \frac{Pb^2}{L^3}(3a+b) $$
$$ V_{FB} = \frac{Pa^2}{L^3}(a+3b) $$

### 2. Uniformly Distributed Load

A uniformly distributed load w per unit length spanning the entire beam length L.

```
    A   w w w w w w w w   B
    |←——————————————————→|
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = -\frac{wL^2}{12} $$
$$ M_{FB} = \frac{wL^2}{12} $$

#### Fixed-End Shears:
$$ V_{FA} = \frac{wL}{2} $$
$$ V_{FB} = \frac{wL}{2} $$

### 3. Partially Distributed Uniform Load

A uniformly distributed load w per unit length applied over a portion of the beam from distance c to d from end A.

```
    A       w w w w       B
    |---c-->|-----|<--e---|
    |<--------L---------->|
```

Where e = L - (c + (d-c)) = L - d

#### Fixed-End Moments:
$$ M_{FA} = -\frac{w}{L^2}[(d-c)(L^2 - (d-c)^2 - 2Lc)] $$
$$ M_{FB} = \frac{w}{L^2}[(d-c)(L^2 - (d-c)^2 - 2Le)] $$

#### Fixed-End Shears:
$$ V_{FA} = \frac{w(d-c)}{2L^3}[2Lc(L-c) + (d-c)(2L-d-c)] $$
$$ V_{FB} = \frac{w(d-c)}{2L^3}[2Le(L-e) + (d-c)(2L-d-c)] $$

### 4. Linearly Varying Load

A linearly varying load starting with intensity w₁ at end A and intensity w₂ at end B.

```
    A  w₁                 B
    |  |                w₂|
    |  v                 v|
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = -\frac{L^2}{30}(2w_1 + w_2) $$
$$ M_{FB} = \frac{L^2}{30}(w_1 + 2w_2) $$

#### Fixed-End Shears:
$$ V_{FA} = \frac{L}{20}(7w_1 + 3w_2) $$
$$ V_{FB} = \frac{L}{20}(3w_1 + 7w_2) $$

### 5. Concentrated Moment

A concentrated moment M applied at a distance a from end A and b from end B (L = a + b).

```
    A         M           B
    |----a--->↻<---b------|
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = -\frac{Mb^2}{L^2} $$
$$ M_{FB} = -\frac{Ma^2}{L^2} $$

#### Fixed-End Shears:
$$ V_{FA} = -\frac{6Mab}{L^3} $$
$$ V_{FB} = \frac{6Mab}{L^3} $$

### 6. Temperature Gradient

A temperature differential ΔT between top and bottom fibers of the beam with depth h and coefficient of thermal expansion α.

```
    A                     B
    |---------------------|
    |      ΔT (top-bot)   |
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = \frac{E\alpha I \Delta T}{h} $$
$$ M_{FB} = \frac{E\alpha I \Delta T}{h} $$

Where:
- E = modulus of elasticity
- I = moment of inertia
- α = coefficient of thermal expansion
- h = depth of the beam

#### Fixed-End Shears:
$$ V_{FA} = 0 $$
$$ V_{FB} = 0 $$

### 7. Support Settlement

A vertical settlement Δ of support B relative to support A.

```
    A                     B
    |---------------------| ↓ Δ
    |<--------L---------->|
```

#### Fixed-End Moments:
$$ M_{FA} = \frac{6EI\Delta}{L^2} $$
$$ M_{FB} = -\frac{6EI\Delta}{L^2} $$

#### Fixed-End Shears:
$$ V_{FA} = \frac{12EI\Delta}{L^3} $$
$$ V_{FB} = -\frac{12EI\Delta}{L^3} $$

## Application in Matrix Structural Analysis

In the direct stiffness method, these fixed-end forces are used to:

1. Calculate the member end forces for members under various loading conditions
2. Convert these member loads to equivalent joint loads
3. Apply these equivalent joint loads in the global structural analysis

The fixed-end forces are transformed from the local coordinate system to the global coordinate system and then applied as equivalent nodal loads in the global stiffness equation:

$$ \{F\} = [K]\{\delta\} + \{F_{FEF}\} $$

Where:
- {F} = vector of external joint loads
- [K] = global stiffness matrix
- {δ} = vector of joint displacements
- {F_FEF} = vector of fixed-end forces transformed to global coordinates