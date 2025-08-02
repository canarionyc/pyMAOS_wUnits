from signal import pause

from pyMAOS.node2d import R2Node
from pyMAOS.elements import R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.structure2d as R2Struct
from pyMAOS.loadcombos import LoadCombo
from pyMAOS.structure2d_plot import plot_structure

# 5 Element Cantilever beam w/ Point Load
loadcase = "D"
loadcombo = LoadCombo("S1", {"D": 1}, ["D"], False, "SLS")
loadcombos = [loadcombo, LoadCombo("D1", {"D": 1.4}, ["D"], False, "ULS")]

# Nodes
N1 = R2Node(1,0, 0)
N2 = R2Node(2,60, 0)
N3 = R2Node(3,120, 0)
N4 = R2Node(4,180, 0)
N5 = R2Node(5,240, 0)
N6 = R2Node(6,300, 0)

N7 = R2Node(7, 0,30)
N8 = R2Node(8, 300, 30)

# Node Restraints
N2.releaseAll()
N3.releaseAll()
N4.releaseAll()
N5.releaseAll()
N6.releaseAll()
N8.releaseAll()

# Node List
nodes = [N1, N2, N3, N4, N5, N6, N7, N8]

# Nodal Loads
N6.loads[loadcase] = [0, -10, 0]
N8.loads[loadcase] = [0, -10, 0]

# Materials
BeamMaterial = Material(1, E=29000, nu=0.3)


# Sections
# W24x55
BeamSection = Section(1,16.2, 1350)

# Members
RF1 = R2Frame(1,N1, N2, BeamMaterial, BeamSection)
RF2 = R2Frame(2,N2, N3, BeamMaterial, BeamSection)
RF3 = R2Frame(3,N3, N4, BeamMaterial, BeamSection)
RF4 = R2Frame(4,N4, N5, BeamMaterial, BeamSection)
RF5 = R2Frame(5,N5, N6, BeamMaterial, BeamSection)
RF6 = R2Frame(6,N7, N8, BeamMaterial, BeamSection)

# Member List
members = [RF1, RF2, RF3, RF4, RF5, RF6]

# Member Release

# Member Loads

# Create the 2D Structure
Structure = R2Struct.R2Structure(nodes, members)
Structure.set_node_uids()
Structure.set_member_uids()

FM = Structure.set_freedom_map()
K = Structure.Kstructure()
U = Structure.solve_linear_static(loadcombo)
Errors = Structure._ERRORS

for combo in loadcombos:
    Structure.solve_linear_static(combo)

# Print Output
print("Errors:")
print(Errors)
print("Displacements:")
for i, node in enumerate(nodes):
    tx = node.displacements[loadcombo.name]
    print(
        f"N{node.uid} -- Ux: {tx[0]:.4E} -- Uy:{tx[1]:.4E} -- Rz:{tx[2]:.4E}"
    )
print("-" * 100)
print("Reactions:")
for i, node in enumerate(nodes):
    rx = node.reactions[loadcombo.name]
    print(
        f"N{node.uid} -- Rx: {rx[0]:.4E} -- Ry:{rx[1]:.4E} -- Mz:{rx[2]:.4E}"
    )
print("-" * 100)
print("Member Forces:")
for i, member in enumerate(members):
    fx = member.end_forces_local[loadcombo.name]

    print(f"M{member.uid}")
    print(
        f"    i -- Axial: {fx[0,0]:.4E} -- Shear: {fx[1,0]:.4E} -- Moment: {fx[2,0]:.4E}"
    )
    print(
        f"    j -- Axial: {fx[3,0]:.4E} -- Shear: {fx[4,0]:.4E} -- Moment: {fx[5,0]:.4E}"
    )

# Max/Min Mz
for member in members:
    print(member.Mzextremes(loadcombo))

# Plot the structure
scaling = {
        "axial_load": 10,
        "normal_load": 10,
        "point_load": 1,
        "axial": 1,
        "shear": 1,
        "moment": 0.01,
        "rotation": 1000,
        "displacement": 50,
    }

plot_structure(nodes, members, loadcombo, scaling)
pause(-1)