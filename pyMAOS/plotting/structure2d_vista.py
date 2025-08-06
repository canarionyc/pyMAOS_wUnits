import pyvista as pv
import numpy as np

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
    input_dir : str, optional
        Directory containing the scaling.json file
    """
    import vtk
    import pyvista as pv
    import numpy as np
    import os
    from pyMAOS.plotting.scaling import get_scaling_from_config

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
    # print(f"Structure type: {type(structure)}")
    print(f"Number of nodes: {len(structure.nodes)}")
    print(f"Number of members: {len(structure.members)}")

    # Find node 1 and calculate offset to place it at x=0
    x_offset = 0
    for node in structure.nodes:
        if node.uid == 1:
            x_offset = node.x.magnitude
            print(f"Found node 1 at x={x_offset}, will offset all nodes to place it at x=0")
            break

    # Create points and cells
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add points for nodes with offset applied
    node_indices = {}  # Map node UIDs to their index in the points array
    for i, node in enumerate(structure.nodes):
        # Extract magnitude values from the Quantity objects and apply offset
        x = node.x.magnitude - x_offset
        y = node.y.magnitude
        print(f"Node {node.uid}: Original ({node.x.magnitude}, {node.y.magnitude}), Adjusted ({x}, {y})")
        points.InsertNextPoint(x, y, 0)
        node_indices[node.uid] = i

    # Add lines for members
    for member in structure.members:
        line = vtk.vtkLine()
        # Get indices based on node UIDs
        i_index = node_indices[member.inode.uid]
        j_index = node_indices[member.jnode.uid]
        line.GetPointIds().SetId(0, i_index)
        line.GetPointIds().SetId(1, j_index)
        cells.InsertNextCell(line)
        print(f"Member: node {member.inode.uid} to {member.jnode.uid}")

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

            # Add deformed points
            for i, node in enumerate(structure.nodes):
                # Get displacements for this node
                dx = node.get_displacement(loadcombo, 'DX').magnitude * displacement_scale if hasattr(node, 'get_displacement') else 0
                dy = node.get_displacement(loadcombo, 'DY').magnitude * displacement_scale if hasattr(node, 'get_displacement') else 0
                rot = node.get_displacement(loadcombo, 'RZ').magnitude * rotation_scale if hasattr(node, 'get_displacement') else 0

                # Calculate deformed position
                x_def = (node.x.magnitude - x_offset) + dx
                y_def = node.y.magnitude + dy

                print(f"Node {node.uid} Deformed: ({x_def}, {y_def}), Displacements: dx={dx}, dy={dy}, rot={rot}")
                deformed_points.InsertNextPoint(x_def, y_def, 0)

            # Add lines for members in deformed shape
            # Add lines for members in deformed shape
            for member in structure.members:
                # Get loads for this member from the loadcombo
                member_loads = []

                if loadcombo is not None:
                    # Extract loads that apply to this member
                    for load in loadcombo.loads:
                        if hasattr(load, 'member_uid') and load.member_uid == member.uid:
                            member_loads.append(load)
                            print(f"Found load for member {member.uid}: {type(load).__name__}")

                # Use the member and loads to generate deformation curve points
                if member_loads:
                    # Generate points along the member for curved deformation
                    num_points = 20
                    member_length = member.length.magnitude
                    x_local = np.linspace(0, member_length, num_points)

                    # Calculate transformation from local to global coordinates
                    dx = member.jnode.x.magnitude - member.inode.x.magnitude
                    dy = member.jnode.y.magnitude - member.inode.y.magnitude
                    length = np.sqrt(dx**2 + dy**2)
                    cos_theta = dx / length
                    sin_theta = dy / length

                    # Create polyline for deformed shape
                    polyline = vtk.vtkPolyLine()
                    polyline.GetPointIds().SetNumberOfIds(num_points)

                    for i, x in enumerate(x_local):
                        # Get displacement value from member's polynomial function
                        # This assumes the member has a method to evaluate deformation at a point
                        y_local = 0  # This should come from the polynomial function

                        for load in member_loads:
                            if hasattr(load, 'Dy2') and load.Dy2 is not None:
                                try:
                                    y_local += load.Dy2.evaluate(x) * displacement_scale
                                except Exception as e:
                                    print(f"Error evaluating displacement: {e}")

                        # Transform to global coordinates with offset
                        x_global = member.inode.x.magnitude - x_offset + x * cos_theta - y_local * sin_theta
                        y_global = member.inode.y.magnitude + x * sin_theta + y_local * cos_theta

                        # Add point to deformed shape
                        point_id = deformed_points.InsertNextPoint(x_global, y_global, 0)
                        polyline.GetPointIds().SetId(i, point_id)

                    deformed_cells.InsertNextCell(polyline)
                else:
                    # If no loads, use straight line for this member
                    line = vtk.vtkLine()
                    i_index = node_indices[member.inode.uid]
                    j_index = node_indices[member.jnode.uid]
                    line.GetPointIds().SetId(0, i_index)
                    line.GetPointIds().SetId(1, j_index)
                    deformed_cells.InsertNextCell(line)

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