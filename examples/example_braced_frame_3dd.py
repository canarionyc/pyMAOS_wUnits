#%% setup
import sys
from matplotlib import pyplot as plt
import numpy as np;    
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=sys.maxsize)
from pprint import pprint
import os
from contextlib import redirect_stdout
import argparse

import matplotlib
matplotlib.use('QtAgg')

from context import pyMAOS

from pyMAOS.nodes import R2Node
from pyMAOS.elements import R2Truss, R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo
from pyMAOS.plot_structure import plot_structure, plot_structure_simple, plot_element, plot_structure_vispy
import csv
import ast
import datetime

print(sys.argv)

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Run a structural analysis using pyMAOS.")
parser.add_argument('-i', '--input-dir', type=str, default='input',
                    help='Directory containing input CSV files (default: input).')
parser.add_argument('-o', '--output-dir', type=str, default='output',
                    help='Directory to save output files (default: output).')
parser.add_argument('-r', '--redirect', action='store_true',
                    help='Redirect output to file.')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Enable debug mode for verbose output.')
args = parser.parse_args()
print(args)
# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Plot the structure
scaling = {
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    }

if args.redirect:
    # Open the file with line buffering (buffering=1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"example_braced_frame_3dd_{timestamp}.txt"
    output_log_path = os.path.join(args.output_dir, log_filename)
    f = open(output_log_path, 'w', buffering=1)
    sys.stdout = f
else:
    f= sys.stdout

# Now all print statements will go to the file and flush on every line
print("This will be written and flushed immediately.")
print(f"Input directory: {args.input_dir}")
print(f"Output directory: {args.output_dir}")
print(f"Debug mode: {args.debug}")

# Sloped beam to test braced frame
loadcase = "D"
loadcombo = LoadCombo("S1", {"D": 1}, ["D"], False, "SLS")

# --- Read nodes ---
nodes_dict = {}
with open(os.path.join(args.input_dir, 'nodes.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True,)
    for row in reader:
        uid = int(row['uid'])
        x = float(row['X'])
        y = float(row['Y'])
        rx = int(row['rx'])
        ry = int(row['ry'])
        rz = int(row['rz'])
        node = R2Node(uid, x, y)
        node.restraints = [rx, ry, rz]
        nodes_dict[uid] = node

if args.debug:
    print("nodes_dict contents:")
    for uid, node in nodes_dict.items():
        print(f"UID: {uid}, x: {node.x}, y: {node.y}, restraints: {node.restraints}")

# If you need a list of nodes in uid order:
node_list = [nodes_dict[uid] for uid in sorted(nodes_dict)]

# --- Read materials ---
materials = {}
with open(os.path.join(args.input_dir, 'Materials.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        uid = int(row['uid'])
        materials[uid] = Material(
            float(row['density']),
            float(row['E'])
#           ,float(row.get('nu', 0.3))
        )

# Example usage:
# To access steel's E: materials[1]['E']

# --- Read sections ---
sections_dict = {}
with open(os.path.join(args.input_dir, 'Sections.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        uid = int(row['uid'])
        sections_dict[uid] = Section(
            float(row['Area']),
            float(row['Ixx']),
            float(row['Iyy'])
        )

# --- Read members ---
element_dict = {}
with open(os.path.join(args.input_dir, 'Elements.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        elem_uid = int(row['elem_uid'])
        n1 = int(row['nodei_uid'])
        n2 = int(row['nodej_uid'])
        mat_uid = int(row['BeamMaterial_uid'])
        sec_uid = int(row['BeamSection_uid'])
        elem_type = row['type'].strip().upper()

        # Foreign key check for material
        if mat_uid not in materials:
            raise ValueError(f"Material uid {mat_uid} in Elements.csv (element {elem_uid}) does not exist in Materials.csv")

        if sec_uid not in sections_dict:
            raise ValueError(f"Section uid {sec_uid} in Elements.csv (element {elem_uid}) does not exist in Sections.csv")

        if elem_type == 'T':
            element = R2Truss(
                uid=elem_uid,
                inode=nodes_dict[n1],
                jnode=nodes_dict[n2],
                material=materials[mat_uid],
                section=sections_dict[sec_uid]
            )
        else:
            element = R2Frame(
                uid=elem_uid,
                inode=nodes_dict[n1],
                jnode=nodes_dict[n2],
                material=materials[mat_uid],
                section=sections_dict[sec_uid]
            )
        element_dict[elem_uid] = element

# If you need a list of elements in uid order for further processing:
element_list = [element_dict[uid] for uid in sorted(element_dict)]



fig, ax = plot_structure_simple(node_list, element_list, loadcombo, scaling)
fig.show()
plt.savefig(os.path.join(args.output_dir, "example_braced_frame_3dd.png"), dpi=300, bbox_inches='tight')

if args.debug:
    print("Detailed element_list:")
    for element in element_list:
        if element.type == "TRUSS":
            print(f"Truss UID: {element.uid}")
            print(f"  In-Node: UID {element.inode.uid} at ({element.inode.x}, {element.inode.y})")
            print(f"  J-Node: UID {element.jnode.uid} at ({element.jnode.x}, {element.jnode.y})")
            print(f"  Material: E = {element.material.E}")
            print(f"  Section: Area = {element.section.Area}")
        elif element.type == "FRAME":
            print(f"Frame UID: {element.uid}")
            print(f"  In-Node: UID {element.inode.uid} at ({element.inode.x}, {element.inode.y})")
            print(f"  J-Node: UID {element.jnode.uid} at ({element.jnode.x}, {element.jnode.y})")
            print(f"  Hinges: {element.hinges}")
            print(f"  Material: E = {element.material.E}")
            print(f"  Section: Area = {element.section.Area}, Ixx = {element.section.Ixx}")
        else:
            print(f"Element UID: {element.uid}, Type: {element.type}")
        print("-" * 50)
        # fig, ax=plot_element(element, loadcombo, scaling)
        # fig.show()
     

N1, N2, N3, N4 = node_list

# Node Restraints
# N1.restraints = [1, 1, 0]
# N2.restraints = [0, 0, 0]
# N3.restraints = [0, 0, 0]
# N4.restraints = [1, 1, 0]

# Nodal Loads
# N2.loads[loadcase] = [50, 0, 0]

# RF2=element_dict[2]  # R2Frame(N2, N3, SteelMaterial, BeamSection)
# # Member Release
# RF2.hinge_i()
# RF2.hinge_j()

# Member Loads
# RF1.add_distributed_load(
#     1 / 12, 1 / 12, 0, 100, loadcase, "X", location_percent=True
# )
# RF2.add_distributed_load(
#     -1 / 12, -1 / 12, 0, 100, loadcase, "yy", location_percent=True
# )
# RF3.add_distributed_load(
#     0.5 / 12, 0.5 / 12, 0, 100, loadcase, "X", location_percent=True
# )

with open(os.path.join(args.input_dir, 'Loads.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        elem_id = int(row['element_id'])
        wi = eval(row['wi'])
        wj = eval(row['wj'])
        a = float(row['a'])
        b = float(row['b'])
        case = row['case']
        direction = row['direction']
        location_percent = row['location_percent'].strip().lower() in ('yes', 'true', '1')
        element = element_dict[elem_id]
        element.add_distributed_load(
            wi, wj, a, b, case, direction, location_percent=location_percent
        )

# Create the 2D Structure
Structure = R2Struct.R2Structure(node_list, element_list)
# Structure.set_node_uids()
# Structure.set_member_uids()


canvas = plot_structure_vispy(node_list, element_list)
import sys
if sys.flags.interactive == 0:
    canvas.app.run()

# The existing Matplotlib plotting is preserved below if you wish to switch back.
# fig, ax = plot_structure_simple(node_list, element_list, loadcombo, scaling)
# fig.show()
# plt.savefig(os.path.join(args.output_dir, "example_braced_frame_3dd.png"), dpi=300, bbox_inches='tight')



# FM = Structure.freedom_map(); 
# K = Structure.Kstructure(FM)

U = Structure.solve_linear_static(loadcombo, **vars(args))
Errors = Structure._ERRORS
print(U)

# Print Output
if len(Errors)>0:
    print("Errors:")
    print(Errors)
print("Displacements:")
for i, node in enumerate(node_list):
    tx = node.displacements[loadcombo.name]
    print(f"N{node.uid} -- Ux: {tx[0]:.4E} -- Uy:{tx[1]:.4E} -- Rz:{tx[2]:.4E}")
f.flush()
print("-" * 100)
print("Reactions:")
for i, node in enumerate(node_list):
    rx = node.reactions[loadcombo.name]
    print(f"N{node.uid} -- Rx: {rx[0]:.4E} -- Ry:{rx[1]:.4E} -- Mz:{rx[2]:.4E}")
f.flush(); print("-" * 100)
print("Member Forces:")
for i, element in enumerate(element_list):
    fx = element.end_forces_local[loadcombo.name]

    print(f"M{element.uid}")
    print(f"    i -- Axial: {fx[0,0]:.4E} -- Shear: {fx[1,0]:.4E} -- Moment: {fx[2,0]:.4E}")
    print(f"    j -- Axial: {fx[3,0]:.4E} -- Shear: {fx[4,0]:.4E} -- Moment: {fx[5,0]:.4E}")
f.flush()

fig, ax = plot_structure(node_list, element_list, loadcombo, scaling)
fig.show()
plt.savefig(os.path.join(args.output_dir, "example_braced_frame_3dd_forces.png"), dpi=300, bbox_inches='tight')


# When done, restore sys.stdout and close the file
sys.stdout = sys.__stdout__
f.close()

