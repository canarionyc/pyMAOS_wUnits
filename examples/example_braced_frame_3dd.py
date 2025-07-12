# -*- coding: utf-8 -*-
import sys
import os
from tkinter import NO
import numpy as np
import json  # Added for reading configuration file
np.set_printoptions(precision=2, suppress=True, linewidth=np.nan, threshold=sys.maxsize)

from contextlib import redirect_stdout
import argparse

import matplotlib

matplotlib.use("QtAgg")

from context import pyMAOS

from pyMAOS.nodes import R2Node, get_nodes_from_csv
from pyMAOS.elements import R2Truss, R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo

import pandas as pd

print(sys.argv)

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Run a structural analysis using pyMAOS.")
parser.add_argument(
    "-i",
    "--input-dir",
    type=str,
    default="input",
    help="Directory containing input CSV files (default: input).",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="output",
    help="Directory to save output files (default: output).",
)

parser.add_argument(
    "-d",
    "--database-dir",
    type=str,
    default="database",
    help="Directory to with beam cross sections and material properties information (default: database).",
)

parser.add_argument(
    "-c",
    "--config-file",
    type=str,
    default="config/scaling.json",
    help="Path to scaling configuration file (default: config/scaling.json).",
)

parser.add_argument(
    "-r", "--redirect", action="store_true", help="Redirect output to file."
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose mode for verbose output."
)
args = parser.parse_args()
print(args)
# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

if args.redirect:
    # Open the file with line buffering (buffering=1)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"example_braced_frame_3dd_{timestamp}.txt"
    output_log_path = os.path.join(args.output_dir, log_filename)
    f = open(output_log_path, "w", buffering=1)
    sys.stdout = f
else:
    f = sys.stdout

# Now all print statements will go to the file and flush on every line
print("This will be written and flushed immediately.")
print(f"Input directory: {args.input_dir}")
print(f"Output directory: {args.output_dir}")
print(f"Debug mode: {args.verbose}")

# Sloped beam to test braced frame
loadcase = "D"
loadcombo = LoadCombo("S1", {"D": 1}, ["D"], False, "SLS")
print(loadcombo)

# This creates a serviceability load combination called "S1" that includes only dead load (D) with a factor of 1.0.
# 4.	Purpose:
# •	Ensures structures are designed for realistic worst-case scenarios
# •	Allows analysis of multiple loading conditions in one solution
# •	Organizes results by loading scenario
# •	Complies with building code requirements
# The load combination determines which loads get applied and with what factors when Structure.solve_linear_static(loadcombo) is called.

# Print loadcombo information
# print("\n--- Load Combination Details ---")
# print(f"Name: {loadcombo.name}")
# print(f"Factors: {loadcombo.factors}")
# print(f"Type: {loadcombo.type}")
# print(f"Is Ultimate: {loadcombo.isUltimate}")
# print("----------------------------\n")

nodes_dict = get_nodes_from_csv(os.path.join(args.input_dir, "nodes.csv"))

if args.verbose:
    print("nodes_dict contents:")
    for uid, node in nodes_dict.items():
        print(f"UID: {uid}, x: {node.x}, y: {node.y}, restraints: {node.restraints}")

# Add node loads from NodeLoads.csv
try:
    node_loads_csv = os.path.join(args.input_dir, "NodeLoads.csv")
    print(f"Reading node loads from {node_loads_csv}...")
    
    # Read with pandas instead of csv module
    df_node_loads = pd.read_csv(
        node_loads_csv,
        skipinitialspace=True,
        skip_blank_lines=True
    )
    
    # Clean whitespace in string columns
    for col in df_node_loads.select_dtypes(include=['object']):
        df_node_loads[col] = df_node_loads[col].str.strip()
    
    # Group by node_id (since multiple loads can be applied to same node)
    for node_id, group in df_node_loads.groupby('node_id'):
        node_id = int(node_id)
        
        # Find the node in the dictionary
        if node_id in nodes_dict:
            node = nodes_dict[node_id]
            
            # Apply all loads for this node
            for _, row in group.iterrows():
                loadcase = row["loadcase"] if "loadcase" in row else "D"
                fx = float(row["Fx"])
                fy = float(row["Fy"])
                mz = float(row["Mz"])
                
                # Apply the load to the node
                node.add_nodal_load(fx, fy, mz, loadcase)
                if args.verbose:
                    print(f"Added load to Node {node_id}: Fx={fx}, Fy={fy}, Mz={mz}, Case={loadcase}")
        else:
            print(f"Warning: Node {node_id} from NodeLoads.csv not found in nodes_dict")
    
    print(f"Successfully loaded {len(df_node_loads)} node loads")
    
except FileNotFoundError:
    print(f"NodeLoads.csv not found in {args.input_dir}. No nodal loads applied.")
except Exception as e:
    print(f"Error reading node loads: {e}")


# If you need a list of nodes in uid order:
node_list = [nodes_dict[uid] for uid in sorted(nodes_dict)]

for node in node_list:
    print(node)
    node.display_loads()  # Display loads for each node

from pyMAOS.database import get_materials_from_csv, get_sections_from_csv
get_materials_from_csv = pyMAOS.database.get_materials_from_csv
get_sections_from_csv = pyMAOS.database.get_sections_from_csv   
# --- Read materials and sections ---
# Read materials and sections from CSV files
materials_csv= os.path.join(args.database_dir, "MaterialsDb.csv")
materials_dict = get_materials_from_csv(materials_csv)
print(materials_dict)


# Read sections from CSV file
sections_csv= os.path.join(args.database_dir, "SectionsDb.csv")
sections_dict = get_sections_from_csv(sections_csv)
print(sections_dict)
# Check if materials and sections were loaded correctly
if not materials_dict:
    raise ValueError(f"No materials found in {materials_csv}. Please check the file.")

# --- Read members ---
element_dict = {}
elements_csv = os.path.join(args.input_dir, "Elements.csv")
print(f"Reading elements from {elements_csv}...")

try:
    # Read elements directly with pandas - handles whitespace and blank lines better
    df_elements = pd.read_csv(
        elements_csv, 
        skipinitialspace=True,  # Skip initial whitespace around delimiters
        skip_blank_lines=True,  # Skip blank lines
        index_col="elem_uid"    # Set elem_uid as index
        ,
        dtype={
            "elem_uid": "Int32",
            "nodei_uid": "Int32",
            "nodej_uid": "Int32",
            "type": "string",
            "mat_uid": "Int32",
            "sec_uid": "Int32"
        }
    )
    
    # Clean up all whitespace in string columns
    for col in df_elements.select_dtypes(include=['object']):
        df_elements[col] = df_elements[col].str.strip()
    
    # Check for duplicate indices that pandas might have handled silently
    if len(df_elements.index) != len(df_elements.index.unique()):
        duplicates = df_elements.index[df_elements.index.duplicated()].unique()
        raise ValueError(f"Duplicate element IDs found: {duplicates}")
    
    print(f"Successfully loaded {len(df_elements)} elements")
    if args.verbose:
        print("Elements DataFrame:")
        print(df_elements.head())
        
except Exception as e:
    print(f"Error reading elements CSV: {e}")
    raise
print(f"Elements DataFrame shape: {df_elements.shape}")

for elem_uid, row in df_elements.iterrows():
    elem_uid = int(elem_uid)
    n1 = int(row["nodei_uid"])
    n2 = int(row["nodej_uid"])
    mat_uid = int(row["mat_uid"])
    sec_uid = int(row["sec_uid"])
    elem_type = str(row["type"]).strip().upper()

    # Foreign key check for material
    if mat_uid not in materials_dict:
        raise ValueError(
            f"Material uid {mat_uid} in {elements_csv} (element {elem_uid}) does not exist in {materials_csv}"
        )

    if sec_uid not in sections_dict:
        raise ValueError(
            f"Section uid {sec_uid} in {elements_csv} (element {elem_uid}) does not exist in {sections_csv}"
        )

    if elem_type == "T":
        element = R2Truss(
            uid=elem_uid,
            inode=nodes_dict[n1],
            jnode=nodes_dict[n2],
            material=materials_dict[mat_uid],
            section=sections_dict[sec_uid],
        )
    else:
        element = R2Frame(
            uid=elem_uid,
            inode=nodes_dict[n1],
            jnode=nodes_dict[n2],
            material=materials_dict[mat_uid],
            section=sections_dict[sec_uid],
        )
    element_dict[elem_uid] = element

# If you need a list of elements in uid order for further processing:
element_list = [element_dict[uid] for uid in sorted(element_dict)]
print(f"Number of elements: {len(element_list)}")

if args.verbose:
    print("Detailed element_list:")
    for element in element_list:
        if element.type == "TRUSS":
            print(f"Truss UID: {element.uid}")
            print(
                f"  In-Node: UID {element.inode.uid} at ({element.inode.x}, {element.inode.y})"
            )
            print(
                f"  J-Node: UID {element.jnode.uid} at ({element.jnode.x}, {element.jnode.y})"
            )
            print(f"  Material: E = {element.material.E}")
            print(f"  Section: Area = {element.section.Area}")
        elif element.type == "FRAME":
            print(f"Frame UID: {element.uid}")
            print(
                f"  In-Node: UID {element.inode.uid} at ({element.inode.x}, {element.inode.y})"
            )
            print(
                f"  J-Node: UID {element.jnode.uid} at ({element.jnode.x}, {element.jnode.y})"
            )
            print(f"  Hinges: {element.hinges}")
            print(f"  Material: E = {element.material.E}")
            print(
                f"  Section: Area = {element.section.Area}, Ixx = {element.section.Ixx}"
            )
        else:
            print(f"Element UID: {element.uid}, Type: {element.type}")
        print("-" * 50)
        # fig, ax=plot_element(element, loadcombo, scaling)
        # fig.show()


# Process element loads using pandas
element_loads_csv = os.path.join(args.input_dir, "ElementLoads.csv")
try:
    df_element_loads = pd.read_csv(element_loads_csv,
                                    skipinitialspace=True,  # Skip initial whitespace around delimiters
        skip_blank_lines=True,  # Skip blank lines
        index_col="load_uid"    # Set elem_uid as index
        )
    
    for load_uid, row in df_element_loads.iterrows():
        elem_id = int(row["elem_id"])
        wi = eval(row["wi"])
        wj = eval(row["wj"])
        a = float(row["a"])
        b = float(row["b"])
        case = row["case"]
        direction = row["direction"]
        location_percent = str(row["location_percent"]).strip().lower() in (
            "yes", "true", "1",
        )
        element = element_dict[elem_id]
        element.add_distributed_load(
            wi, wj, a, b, case, direction, location_percent=location_percent
        )
    if args.verbose:
        print(f"Loaded {len(df_element_loads)} distributed loads from {element_loads_csv}")
except FileNotFoundError:
    print(f"ElementLoads.csv not found in {args.input_dir}. No element loads applied.")

# Create the 2D Structure
Structure = R2Struct.R2Structure(node_list, element_list)
# Structure.set_node_uids()
# Structure.set_member_uids()

FM = Structure.freedom_map(); print(FM)
K = Structure.Kstructure(FM)

U = Structure.solve_linear_static(loadcombo, **vars(args))
Errors = Structure._ERRORS

# Print Output
if len(Errors) > 0:
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
f.flush()
print("-" * 100)
print("Member Forces:")
for i, element in enumerate(element_list):
    fx = element.end_forces_local[loadcombo.name]

    print(f"M{element.uid}")
    print(
        f"    i -- Axial: {fx[0,0]:.4E} -- Shear: {fx[1,0]:.4E} -- Moment: {fx[2,0]:.4E}"
    )
    print(
        f"    j -- Axial: {fx[3,0]:.4E} -- Shear: {fx[4,0]:.4E} -- Moment: {fx[5,0]:.4E}"
    )
f.flush()

# fig, ax = plot_structure(node_list, element_list, loadcombo, scaling)
# fig.show()
# plt.savefig(os.path.join(args.output_dir, "example_braced_frame_3dd_forces.png"), dpi=300, bbox_inches='tight')

# --- Choose a plotting library ---

# Use VTK for plotting
from pyMAOS.plot_structure import get_scaling_from_config, plot_structure_loadcombos_vtk, plot_structure_vtk
# Read scaling parameters from configuration file

scaling = get_scaling_from_config(args.config_file)
print("Using scaling parameters:")
for key, value in scaling.items():
    print(f"  {key}: {value}")
# Use VisPy for plotting
# canvas = plot_structure_vispy(node_list, element_list, loadcombo, scaling)
# import sys
# if sys.flags.interactive == 0:
#     canvas.app.run()

# The existing Matplotlib plotting is preserved below if you wish to switch back.
# fig, ax = plot_structure_simple(node_list, element_list, loadcombo, scaling)
# fig.show()
# plt.savefig(os.path.join(args.output_dir, "example_braced_frame_3dd.png"), dpi=300, bbox_inches='tight')


plot_structure_vtk(node_list, element_list, loadcombo, scaling)
plot_structure_loadcombos_vtk(node_list, element_list, loadcombo, scaling)
# When done, restore sys.stdout and close the file
sys.stdout = sys.__stdout__
f.close()
