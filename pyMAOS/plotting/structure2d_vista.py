import pyvista as pv
import numpy as np
from pyMAOS.quantity_utils import convert_registry

import traceback

def plot_structure_pv(structure, loadcombo=None, scale=1.0, scaling_file=None):
    """
    Plot the structure using PyVista/VTK with scaling from config file.

    Parameters
    ----------
    structure : R2Structure
        Structure object containing nodes and members
    loadcombo : LoadCombo, optional
        Load combination for plotting deformed shape
    scale : float, optional
        Scale factor for deformations (default=1.0)
    scaling_file : str, optional
        Path to the scaling.json file
    """
    import vtk
    import pyvista as pv
    import numpy as np
    import os
    import pyMAOS
    from pyMAOS.plotting.scaling import get_scaling_from_config
    from pyMAOS.quantity_utils import convert_registry
    from pyMAOS import unit_manager
    # Ensure we're using the global registry for all quantities in this function
    ureg = pyMAOS.unit_manager.ureg

    # Find scaling.json in input directory or current directory
    if scaling_file is None and os.path.exists('scaling.json'):
        scaling_file = 'scaling.json'

    # Get scaling parameters from config file or use defaults
    if scaling_file and os.path.exists(scaling_file):
        print(f"Loading scaling parameters from {scaling_file}")
        scaling_params = get_scaling_from_config(scaling_file)
        rotation_scale = scaling_params.get("rotation", 5000)
        displacement_scale = scaling_params.get("displacement", 100) * scale
        print(f"Using rotation_scale={rotation_scale}, displacement_scale={displacement_scale}")
    else:
        print("No scaling.json found, using default values")
        rotation_scale = 5000
        displacement_scale = 100 * scale

    # Debug info
    print(f"Number of nodes: {len(structure.nodes)}")
    print(f"Number of members: {len(structure.members)}")

    # Check for registry consistency
    registry_mismatch = False
    for node in structure.nodes:
        if hasattr(node.x, '_REGISTRY') and node.x._REGISTRY is not ureg:
            registry_mismatch = True
            print(f"WARNING: Node {node.uid} uses registry {id(node.x._REGISTRY)} instead of global registry {id(ureg)}")
            break

    if registry_mismatch:
        print("Converting all quantities to global registry...")
        from pyMAOS.quantity_utils import convert_all_quantities
        structure = convert_all_quantities(structure, ureg)

    # Find node 1 and calculate offset to place it at x=0
    x_offset = 0
    for node in structure.nodes:
        if node.uid == 1:
            # Ensure this quantity uses the global registry
            x = convert_registry(node.x, ureg)
            x_offset = x.magnitude
            print(f"Found node 1 at x={x_offset}, will offset all nodes to place it at x=0")
            break

    # Create points and cells
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add points for nodes with offset applied
    node_indices = {}  # Map node UIDs to their index in the points array
    for i, node in enumerate(structure.nodes):
        # Extract magnitude values from the Quantity objects and apply offset
        x = convert_registry(node.x, ureg).magnitude - x_offset
        y = convert_registry(node.y, ureg).magnitude
        print(f"Node {node.uid}: Original ({node.x.magnitude}, {node.y.magnitude}), Adjusted ({x}, {y})")
        points.InsertNextPoint(x, y, 0)
        node_indices[node.uid] = i

    # Add lines for members
    for member in structure.members:
        print(f"Member: node {member.inode.uid} to {member.jnode.uid}")
        line = vtk.vtkLine()
        # Get indices based on node UIDs
        i_index = node_indices[member.inode.uid]
        j_index = node_indices[member.jnode.uid]
        line.GetPointIds().SetId(0, i_index)
        line.GetPointIds().SetId(1, j_index)
        cells.InsertNextCell(line)

    # Create a polydata to store points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    # Convert to PyVista mesh for easier handling
    mesh = pv.PolyData(polydata)
    print(mesh)
    # Create PyVista plotter
    plotter = pv.Plotter()

    # Add structure to plot
    plotter.add_mesh(mesh, color='black', line_width=3)

    # Add node markers
    node_size = 0.05  # Adjust as needed
    for node in structure.nodes:
        adjusted_x = node.x.magnitude - x_offset
        plotter.add_mesh(pv.Sphere(radius=node_size, center=(adjusted_x, node.y.magnitude, 0)),
                         color='blue')

    # Create node labels (carefully ensuring non-empty strings)
    if structure.nodes:
        node_points = []
        node_labels = []
        for node in structure.nodes:
            adjusted_x = node.x.magnitude - x_offset
            node_label = f"N{node.uid}"
            # Only add points with valid labels
            if node_label:  # Check that label is not empty
                node_points.append([adjusted_x, node.y.magnitude, 0])
                node_labels.append(node_label)

        # Add labels to plot if we have valid points and labels
        if node_points and node_labels:
            plotter.add_point_labels(
                node_points, node_labels,
                font_size=10,
                point_color='blue',
                text_color='black',
                always_visible=True
            )

        # Plot deformed shape if loadcombo is provided
        if loadcombo is not None:
            # Create points and cells for deformed shape
            deformed_points = vtk.vtkPoints()
            deformed_cells = vtk.vtkCellArray()

            # Add deformed points - using the node's x_displaced and y_displaced methods
            # similar to how it's done in the matplotlib implementation
            for i, node in enumerate(structure.nodes):
                print(f"Node {node.uid}: {node.x.magnitude}, {node.y.magnitude}")
                try:
                    # Use the node's built-in displacement methods which properly handle all transformations
                    if hasattr(node, 'x_displaced') and hasattr(node, 'y_displaced'):
                        # Ensure these quantities use the global registry
                        x_def = convert_registry(node.x_displaced(loadcombo, displacement_scale), ureg).magnitude - x_offset
                        y_def = convert_registry(node.y_displaced(loadcombo, displacement_scale), ureg).magnitude
                    else:
                        # Fallback to direct calculation if methods not available
                        dx = convert_registry(node.get_displacement(loadcombo, 'DX'), ureg).magnitude * displacement_scale if hasattr(node, 'get_displacement') else 0
                        dy = convert_registry(node.get_displacement(loadcombo, 'DY'), ureg).magnitude * displacement_scale if hasattr(node, 'get_displacement') else 0
                        x_def = (convert_registry(node.x, ureg).magnitude - x_offset) + dx
                        y_def = convert_registry(node.y, ureg).magnitude + dy

                    print(f"Node {node.uid} Deformed: ({x_def}, {y_def}) in {unit_manager.get_internal_unit('length')}")
                    deformed_points.InsertNextPoint(x_def, y_def, 0)
                except Exception as e:
                    print(f"Error computing deformed position for node {node.uid}: {e}")
                    # Use undeformed position as fallback
                    x_def = convert_registry(node.x, ureg).magnitude - x_offset
                    y_def = convert_registry(node.y, ureg).magnitude
                    deformed_points.InsertNextPoint(x_def, y_def, 0)

            # Add lines for deformed members
            for member in structure.members:
                i_index = node_indices[member.inode.uid]
                j_index = node_indices[member.jnode.uid]

                # First create basic line between deformed nodes
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i_index)
                line.GetPointIds().SetId(1, j_index)
                deformed_cells.InsertNextCell(line)

                # For frame elements, generate a more accurate deformation curve
                if member.type != "TRUSS" and hasattr(member, 'dglobal_span'):
                    try:
                        # Get deformed shape (global displacements) using the member's method
                        dglobal = member.dglobal_span(loadcombo, displacement_scale)
                        print(f"member {member.uid} dglobal: {dglobal}")
                        # Create a polyline for this member's deformation curve
                        polyline = vtk.vtkPolyLine()
                        num_points = len(dglobal)
                        polyline.GetPointIds().SetNumberOfIds(num_points)

                        # Add all points of the deformation curve
                        for i, point in enumerate(dglobal):
                            # Convert to global coordinates with offset
                            x_global = point[0] + member.inode.x.magnitude - x_offset
                            y_global = point[1] + member.inode.y.magnitude

                            # Add point to deformed shape
                            point_id = deformed_points.InsertNextPoint(x_global, y_global, 0)
                            polyline.GetPointIds().SetId(i, point_id)

                        deformed_cells.InsertNextCell(polyline)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"Error plotting deformed shape for member {member.uid}: {e}")

            # Create polydata for deformed shape
            deformed_polydata = vtk.vtkPolyData()
            deformed_polydata.SetPoints(deformed_points)
            deformed_polydata.SetLines(deformed_cells)

            # Convert to PyVista mesh
            deformed_mesh = pv.PolyData(deformed_polydata)

            # Add deformed shape to plot
            plotter.add_mesh(deformed_mesh, color='red', line_width=2, style='wireframe')

        # Set up camera and view
        plotter.camera_position = 'xy'  # Top-down view
        plotter.enable_parallel_projection()

        # Set background color
        plotter.background_color = 'white'

        # Add a title
        plotter.add_title(f"Structure Plot - {'Deformed' if loadcombo else 'Undeformed'}")

        # Show plot
        plotter.show()