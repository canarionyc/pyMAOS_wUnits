import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Function to read scaling parameters from configuration file
def get_scaling_from_config(config_file_path):
    # Default scaling values
    default_scaling = {
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    }
    import json
    try:
        with open(config_file_path, 'r') as f:
            config_scaling = json.load(f)
            print(f"Loaded scaling configuration from {config_file_path}")
            
            # Update default scaling with values from config file
            for key, value in config_scaling.items():
                if key in default_scaling:
                    default_scaling[key] = value
                else:
                    print(f"Warning: Unknown scaling parameter '{key}' in config file")
    except FileNotFoundError:
        print(f"Scaling configuration file not found: {config_file_path}")
        print("Using default scaling values")
    except json.JSONDecodeError:
        print(f"Error parsing scaling configuration file: {config_file_path}")
        print("Using default scaling values")
    except Exception as e:
        print(f"Error reading scaling configuration: {str(e)}")
        print("Using default scaling values")
    
    return default_scaling


def plot_structure(
    nodes,
    members,
    loadcombo,
    scaling={
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    },
):
    # Plot the structure
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) # Increased figure size for clarity

    axial_loading_scale = scaling.get("axial_load", 1)
    normal_loading_scale = scaling.get("normal_load", 1)
    ptloading_scale = scaling.get("point_load", 1)
    axial_scale = scaling.get("axial", 1)
    shear_scale = scaling.get("shear", 1)
    moment_scale = scaling.get("moment", 1)
    rotation_scale = scaling.get("rotation", 1)
    displace_scale = scaling.get("displacement", 1)
    marker_shape = "o"
    marker_size = 2

    # --- Create a color map for different sections ---
    unique_sections = list(set(member.section for member in members))
    # Using a colormap to get distinct colors
    colors = plt.cm.get_cmap('viridis', len(unique_sections)) 
    section_color_map = {section: colors(i) for i, section in enumerate(unique_sections)}


    axs[0, 0].set_title(
        f"Geometry and Deformed Shape\n scale:{displace_scale}", fontsize=12
    )
    axs[0, 1].set_title(
        f"Member Loading \n axial scale:{axial_loading_scale} \n normal scale:{normal_loading_scale}\n pt load scale:{ptloading_scale}",
        fontsize=12,
    )

    axs[0, 2].set_title(f"Axial Force\n scale:{axial_scale}", fontsize=12)

    axs[1, 0].set_title(f"Shear Force\n scale:{shear_scale}", fontsize=12)

    axs[1, 1].set_title(f"Moment\n scale:{moment_scale}", fontsize=12)

    axs[1, 2].set_title(f"Cross-Section Rotation\n scale:{rotation_scale}", fontsize=12)

    for node in nodes:
        # Plot nodes across all subplots
        for ax in axs.flat:
            ax.plot(node.x, node.y, marker=".", markersize=8, color="red")
        
        # Plot deformed shape on the first subplot
        axs[0, 0].plot(
            node.x_displaced(loadcombo, displace_scale),
            node.y_displaced(loadcombo, displace_scale),
            marker=".",
            markersize=10,
            color="gray",
        )

        # --- Visualize restraints ---
        size = 6  # length of restraint symbol
        if hasattr(node, "restraints"):
            rx, ry, rz = node.restraints
            if rx:
                axs[0, 0].plot([node.x - size, node.x + size], [node.y, node.y], color="blue", linewidth=2)
            if ry:
                axs[0, 0].plot([node.x, node.x], [node.y - size, node.y + size], color="green", linewidth=2)
            if rz:
                theta = np.linspace(0, 2 * np.pi, 100)
                axs[0, 0].plot(node.x + (size/2) * np.cos(theta), node.y + (size/2) * np.sin(theta), color="purple", linewidth=1.5)

    for member in members: 
        member_color = section_color_map.get(member.section, "black") # Default to black if section not in map
        
        # Plot member geometry on all subplots, colored by section on the first one
        for i, ax in enumerate(axs.flat):
            color = member_color if i == 0 else "black" # Color by section only on the geometry plot
            ax.plot(
                [member.inode.x, member.jnode.x],
                [member.inode.y, member.jnode.y],
                linewidth=1.5,
                color=color,
            )

        aglobal = member.Aglobal_plot(loadcombo, axial_scale)
        dglobal = member.Dglobal_plot(loadcombo, displace_scale)

        axs[0, 2].plot(
            (aglobal[:, 0] + member.inode.x),
            (aglobal[:, 1] + member.inode.y),
            linewidth=1,
            color="blue",
            marker=marker_shape,
            markersize=marker_size,
        )

        axs[0, 0].plot(
            (dglobal[:, 0] + member.inode.x),
            (dglobal[:, 1] + member.inode.y),
            linewidth=1,
            color="gray",
            marker=marker_shape,
            markersize=marker_size,
        )

        if member.type != "TRUSS":
            vglobal = member.Vglobal_plot(loadcombo, shear_scale)
            mglobal = member.Mglobal_plot(loadcombo, moment_scale)
            sglobal = member.Sglobal_plot(loadcombo, rotation_scale)
            wxglobal = member.Wxglobal_plot(loadcombo, axial_loading_scale, ptloading_scale)
            wyglobal = member.Wyglobal_plot(loadcombo, normal_loading_scale, ptloading_scale)

            axs[0, 1].plot((wxglobal[:, 0] + member.inode.x), (wxglobal[:, 1] + member.inode.y), linewidth=1, color="blue", marker=marker_shape, markersize=marker_size)
            axs[0, 1].plot((wyglobal[:, 0] + member.inode.x), (wyglobal[:, 1] + member.inode.y), linewidth=1, color="red", marker=marker_shape, markersize=marker_size)
            axs[1, 0].plot((vglobal[:, 0] + member.inode.x), (vglobal[:, 1] + member.inode.y), linewidth=1, color="green", marker=marker_shape, markersize=marker_size)
            axs[1, 1].plot((mglobal[:, 0] + member.inode.x), (mglobal[:, 1] + member.inode.y), linewidth=1, color="red", marker=marker_shape, markersize=marker_size)
            axs[1, 1].plot((member.inode.x, mglobal[0, 0] + member.inode.x), (member.inode.y, mglobal[0, 1] + member.inode.y), linewidth=1, color="red")
            axs[1, 1].plot((member.jnode.x, mglobal[-1, 0] + member.inode.x), (member.jnode.y, mglobal[-1, 1] + member.inode.y), linewidth=1, color="red")
            axs[1, 2].plot((sglobal[:, 0] + member.inode.x), (sglobal[:, 1] + member.inode.y), linewidth=1, color="purple", marker=marker_shape, markersize=marker_size)

    for ax in axs.flat:
        ax.grid(True)
        ax.set_aspect("equal", "box")

    fig.tight_layout()
    return fig, axs


def plot_structure_simple(
    nodes,
    members,
    loadcombo,
    scaling={
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    },
):
    # Plot the structure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes and their labels
    for node in nodes:
        ax.plot(node.x, node.y, marker="o", markersize=6, color="red", label="Node" if "Node" not in ax.get_legend_handles_labels()[1] else "")
        ax.text(node.x, node.y + 5, f'N{node.uid}', color='darkred', fontsize=9, ha='center')

        # Visualize restraints
        if hasattr(node, "restraints"):
            rx, ry, rz = node.restraints
            size = 5  # Length of restraint symbol
            if rx:  # Horizontal restraint
                ax.plot([node.x - size, node.x + size], [node.y, node.y], color="blue", linewidth=2, label="Restraint (Ux)" if "Restraint (Ux)" not in ax.get_legend_handles_labels()[1] else "")
            if ry:  # Vertical restraint
                ax.plot([node.x, node.x], [node.y - size, node.y + size], color="green", linewidth=2, label="Restraint (Uy)" if "Restraint (Uy)" not in ax.get_legend_handles_labels()[1] else "")
            if rz:  # Rotational restraint
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(
                    node.x + (size / 2) * np.cos(theta),
                    node.y + (size / 2) * np.sin(theta),
                    color="purple",
                    linewidth=1.5,
                    label="Restraint (Rz)" if "Restraint (Rz)" not in ax.get_legend_handles_labels()[1] else "",
                )

    # Plot members and their labels
    for member in members:
        ax.plot(
            [member.inode.x, member.jnode.x],
            [member.inode.y, member.jnode.y],
            color="black",
            linewidth=1.5,
            label="Element" if "Element" not in ax.get_legend_handles_labels()[1] else "",
        )
        # Calculate midpoint for the label
        mid_x = (member.inode.x + member.jnode.x) / 2
        mid_y = (member.inode.y + member.jnode.y) / 2
        ax.text(mid_x, mid_y, f'M{member.uid}', color='darkblue', fontsize=9, ha='center', va='center', backgroundcolor=(1,1,1,0.7))


    # Add grid, labels, and legend
    ax.grid(True)
    ax.set_aspect("equal", "box")
    ax.set_title("Structure Visualization", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend()

    # Return the figure and axis for further customization or saving
    return fig, ax

# python pyMAOS/plot_structure.py
def plot_element(element, loadcombo, scaling={
        "axial": 2,
        "displacement": 100,
    }):
    """
    Plots a single element with its nodes, geometry, and (if frame)
    internal force and deformed shape information.
    
    Parameters
    ----------
    element : R2Truss or R2Frame instance
        The structural element to plot.
    loadcombo : LoadCombo
        The load combination to evaluate deformations and forces.
    scaling : dict, optional
        Scaling factors for internal force and deformations.
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the element's initial geometry and nodes.
    ax.plot(element.inode.x, element.inode.y, marker="o", markersize=6, color="red", label="In-Node")
    ax.plot(element.jnode.x, element.jnode.y, marker="s", markersize=6, color="green", label="J-Node")
    ax.plot([element.inode.x, element.jnode.x],
            [element.inode.y, element.jnode.y],
            color="black", linewidth=2, label="Element Geometry")
    
    ax.set_title(f"Element UID: {element.uid} ({element.type})", fontsize=14)
    
    # If the element is a frame, add deformed shape and axial force plot.
    if element.type != "TRUSS" and element._loaded:
        disp_scale = scaling.get("displacement", 100)
        axial_scale = scaling.get("axial", 2)
        
        try:
            # Get deformed shape (global displacements) and overlay it.c v
            dglobal = element.Dglobal_plot(loadcombo, disp_scale)
            ax.plot(dglobal[:, 0] + element.inode.x,
                    dglobal[:, 1] + element.inode.y,
                    linestyle="--", color="gray", label="Deformed Shape")
        except Exception as e:
            print(f"Error plotting deformed shape for element {element.uid}: {e}")
        
        try:
            # Plot axial force distribution (if available)
            aglobal = element.Aglobal_plot(loadcombo, axial_scale)
            ax.plot(aglobal[:, 0] + element.inode.x,
                    aglobal[:, 1] + element.inode.y,
                    linestyle=":", color="blue", label="Axial Force")
        except Exception as e:
            print(f"Error plotting axial force for element {element.uid}: {e}")
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    return fig, ax

def plot_structure_vispy(nodes, members, loadcombo=None, scaling=None):
    """
    Visualizes the structure using VisPy, optimized for performance.

    Parameters
    ----------
    nodes : list
        A list of R2Node objects.
    members : list
        A list of R2Truss or R2Frame objects.
    loadcombo : LoadCombo, optional
        The load combination to display deformed shape.
    scaling : dict, optional
        Scaling factors for deformations.
    """
    from vispy import scene
    from vispy.app import use_app
    from vispy.color import Color, ColorArray
    import numpy as np

    try:
        use_app('pyqt6')
    except Exception:
        print("PyQt6 backend not found, VisPy will attempt to find another.")
        use_app()

    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(800, 600), bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    # --- Member and Line Preparation ---
    # To color lines per-segment, we must duplicate the vertices for each line.
    line_pos = []
    line_colors = []

    # Define specific colors for each member type.
    type_color_map = {
        "FRAME": Color("blue").rgba,
        "TRUSS": Color("green").rgba
    }
    default_color = Color('black').rgba

    for member in members:
        # Add start and end node positions for this member
        line_pos.append([member.inode.x, member.inode.y])
        line_pos.append([member.jnode.x, member.jnode.y])
        
        # Get the color for this member type
        color = type_color_map.get(member.type, default_color)
        
        # Add the color for both vertices of the segment
        line_colors.append(color)
        line_colors.append(color)

    # Convert to NumPy arrays with efficient dtypes
    line_pos_array = np.array(line_pos, dtype=np.float32)
    line_color_array = np.array(line_colors, dtype=np.float32)

    # Create a single Line visual. 'connect' is not needed as vertices are ordered.
    lines = scene.visuals.Line(
        pos=line_pos_array, 
        color=line_color_array,
        width=3, 
        method='gl',
        parent=view.scene
    )

    # --- Node Customization ---
    # The node data is prepared separately from the lines.
    node_pos = np.array([[node.x, node.y] for node in nodes], dtype=np.float32)
    node_colors = []
    for node in nodes:
        if hasattr(node, "restraints") and any(node.restraints):
            node_colors.append(Color('blue').rgba)
        else:
            node_colors.append(Color('red').rgba)

    markers = scene.visuals.Markers(
        pos=node_pos, 
        face_color=ColorArray(node_colors), 
        edge_color='black',
        symbol='disc',
        size=10, 
        parent=view.scene
    )

    # --- Deformed Shape (Example of updating data) ---
    if loadcombo and scaling:
        displace_scale = scaling.get("displacement", 100)
        
        deformed_pos = []
        for member in members:
            deformed_pos.append([member.inode.x_displaced(loadcombo, displace_scale), member.inode.y_displaced(loadcombo, displace_scale)])
            deformed_pos.append([member.jnode.x_displaced(loadcombo, displace_scale), member.jnode.y_displaced(loadcombo, displace_scale)])

        deformed_pos_array = np.array(deformed_pos, dtype=np.float32)
        
        scene.visuals.Line(
            pos=deformed_pos_array, 
            color='gray', 
            width=2, 
            method='gl',
            parent=view.scene
        )

    view.camera.set_range()

    return canvas


def plot_structure_vtk(nodes, members, loadcombo=None, scaling=None):
    """
    Visualizes the structure using the Visualization Toolkit (VTK) with interactivity.
    Includes arrows representing forces applied to nodes.

    Parameters
    ----------
    nodes : list
        A list of R2Node objects.
    members : list
        A list of R2Truss or R2Frame objects.
    loadcombo : LoadCombo, optional
        The load combination to display deformed shape.
    scaling : dict, optional
        Scaling factors for deformations.
    """
    import vtk
    import math
    import numpy as np

    # --- 1. Create VTK data for the original structure ---
    points = vtk.vtkPoints(); points.degug = True
    lines = vtk.vtkCellArray(); lines.debug = True
    line_colors = vtk.vtkUnsignedCharArray()
    line_colors.SetNumberOfComponents(3)
    line_colors.SetName("Colors")

    type_color_map = {"FRAME": (0, 0, 255), "TRUSS": (0, 255, 0)}
    default_color = (0, 0, 0)

    node_uid_to_vtk_id = {node.uid: i for i, node in enumerate(nodes)}
    for node in nodes:
        points.InsertNextPoint(node.x, node.y, 0)

    for member in members:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, node_uid_to_vtk_id[member.inode.uid])
        line.GetPointIds().SetId(1, node_uid_to_vtk_id[member.jnode.uid])
        lines.InsertNextCell(line)
        color = type_color_map.get(member.type, default_color)
        line_colors.InsertNextTuple3(color[0], color[1], color[2])

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    poly_data.GetCellData().SetScalars(line_colors)

    # --- 2. Create Actors for the original structure and node labels ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(3)

    # Node Labels
    node_labels_poly = vtk.vtkPolyData()
    node_labels_poly.SetPoints(points)
    node_labels = vtk.vtkStringArray()
    node_labels.SetName("NodeLabels")
    for node in nodes:
        node_labels.InsertNextValue(f"N{node.uid}")
    node_labels_poly.GetPointData().AddArray(node_labels)
    
    node_label_mapper = vtk.vtkLabeledDataMapper()
    node_label_mapper.SetInputData(node_labels_poly)
    node_label_mapper.SetFieldDataName("NodeLabels")
    node_label_mapper.SetLabelModeToLabelFieldData()
    node_label_mapper.GetLabelTextProperty().SetColor(0.8, 0.1, 0.1)
    node_label_actor = vtk.vtkActor2D()
    node_label_actor.SetMapper(node_label_mapper)

    # --- 3. Create Boxed Member Labels ---
    member_label_actors = []
    for member in members:
        p1 = np.array([member.inode.x, member.inode.y, 0])
        p2 = np.array([member.jnode.x, member.jnode.y, 0])
        
        mid_point = (p1 + p2) / 2
        diff = p2 - p1
        angle_rad = math.atan2(diff[1], diff[0])
        angle_deg = math.degrees(angle_rad)

        # Create the text
        text_source = vtk.vtkVectorText()
        text_source.SetText(f"M{member.uid}")
        text_source.Update()
        
        # Get text bounds
        text_bounds = text_source.GetOutput().GetBounds()
        text_width = text_bounds[1] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[2]
       
        
        # Transform both text and box
        transform = vtk.vtkTransform()
        transform.Translate(mid_point)
        transform.RotateZ(angle_deg)
        transform.Scale(2.5, 2.5, 1)  # Increased scale for better visibility
        
        # Create the text actor with improved visibility
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())
        text_actor = vtk.vtkActor()
        text_actor.SetMapper(text_mapper)
        text_actor.GetProperty().SetColor(0.0, 0.0, 0.6)  # Darker blue text
        # text_actor.GetProperty().SetAmbient(0.5)  # Make text more visible
        text_actor.SetUserTransform(transform)
        
        # Add both actors to the list
        member_label_actors.append(text_actor)

    # --- 3a. Create Hinge Visualizations ---
    hinge_actors = []
    for member in members:
        if hasattr(member, 'hinges') and any(member.hinges):
            # Get member endpoints
            p1 = np.array([member.inode.x, member.inode.y, 0])
            p2 = np.array([member.jnode.x, member.jnode.y, 0])
            
            # Create spheres for each hinge
            if member.hinges[0]:  # i-node hinge
                hinge_sphere = vtk.vtkSphereSource()
                hinge_sphere.SetCenter(p1)
                hinge_sphere.SetRadius(0.5)  # Adjust size as needed
                hinge_sphere.SetPhiResolution(16)
                hinge_sphere.SetThetaResolution(16)
                
                hinge_mapper = vtk.vtkPolyDataMapper()
                hinge_mapper.SetInputConnection(hinge_sphere.GetOutputPort())
                hinge_actor = vtk.vtkActor()
                hinge_actor.SetMapper(hinge_mapper)
                hinge_actor.GetProperty().SetColor(1.0, 0.8, 0.0)  # Gold color
                hinge_actors.append(hinge_actor)
            
            if member.hinges[1]:  # j-node hinge
                hinge_sphere = vtk.vtkSphereSource()
                hinge_sphere.SetCenter(p2)
                hinge_sphere.SetRadius(0.5)  # Adjust size as needed
                hinge_sphere.SetPhiResolution(16)
                hinge_sphere.SetThetaResolution(16)
                
                hinge_mapper = vtk.vtkPolyDataMapper()
                hinge_mapper.SetInputConnection(hinge_sphere.GetOutputPort())
                hinge_actor = vtk.vtkActor()
                hinge_actor.SetMapper(hinge_mapper)
                hinge_actor.GetProperty().SetColor(1.0, 0.8, 0.0)  # Gold color
                hinge_actors.append(hinge_actor)

    # --- 3b. NEW: Add Force Arrows for node loads ---
    force_arrow_actors = []
    if loadcombo:
        # Define force scaling for arrows
        force_scale = scaling.get("point_load", 1) if scaling else 1
        
        for node in nodes:
            # Check if node has loads for any load case in the combination
            has_forces = False
            force_vector = [0, 0, 0]
            
            for load_case, load in node.loads.items():
                load_factor = loadcombo.factors.get(load_case, 0)
                if load_factor != 0:
                    has_forces = True
                    # Scale load by combination factor and add to force vector
                    for i in range(3):  # For each component (Fx, Fy, Mz)
                        force_vector[i] += load[i] * load_factor
            
            if has_forces:
                # Create arrows for horizontal and vertical forces (if significant)
                force_threshold = 0.001  # Minimum force to show an arrow
                
                # Create horizontal force arrow (X direction)
                if abs(force_vector[0]) > force_threshold:
                    arrow_source = vtk.vtkArrowSource()
                    arrow_source.SetTipResolution(16)
                    arrow_source.SetShaftResolution(16)
                    
                    # Scale and position the arrow
                    arrow_length = abs(force_vector[0]) * force_scale
                    arrow_transform = vtk.vtkTransform()
                    arrow_transform.Translate(node.x, node.y, 0)
                    
                    # Rotate based on force direction
                    if force_vector[0] < 0:
                        arrow_transform.RotateZ(180)  # Left pointing arrow
                    
                    # Scale arrow length
                    arrow_transform.Scale(arrow_length, arrow_length, arrow_length)
                    
                    # Create arrow actor
                    arrow_mapper = vtk.vtkPolyDataMapper()
                    arrow_mapper.SetInputConnection(arrow_source.GetOutputPort())
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(arrow_mapper)
                    arrow_actor.SetUserTransform(arrow_transform)
                    arrow_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red for X-force
                    force_arrow_actors.append(arrow_actor)
                
                # Create vertical force arrow (Y direction)
                if abs(force_vector[1]) > force_threshold:
                    arrow_source = vtk.vtkArrowSource()
                    arrow_source.SetTipResolution(16)
                    arrow_source.SetShaftResolution(16)
                    
                    # Scale and position the arrow
                    arrow_length = abs(force_vector[1]) * force_scale
                    arrow_transform = vtk.vtkTransform()
                    arrow_transform.Translate(node.x, node.y, 0)
                    
                    # Rotate based on force direction
                    if force_vector[1] > 0:
                        arrow_transform.RotateZ(90)  # Upward pointing arrow
                    else:
                        arrow_transform.RotateZ(-90)  # Downward pointing arrow
                    
                    # Scale arrow length
                    arrow_transform.Scale(arrow_length, arrow_length, arrow_length)
                    
                    # Create arrow actor
                    arrow_mapper = vtk.vtkPolyDataMapper()
                    arrow_mapper.SetInputConnection(arrow_source.GetOutputPort())
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(arrow_mapper)
                    arrow_actor.SetUserTransform(arrow_transform)
                    arrow_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green for Y-force
                    force_arrow_actors.append(arrow_actor)
                
                # Create moment visualization (Z direction)
                if abs(force_vector[2]) > force_threshold:
                    # For moments, create a circle with an arrow
                    radius = 1.0 * force_scale
                    resolution = 20
                    
                    # Create circle source
                    circle_source = vtk.vtkRegularPolygonSource()
                    circle_source.SetNumberOfSides(resolution)
                    circle_source.SetRadius(radius)
                    circle_source.SetCenter(node.x, node.y, 0)
                    
                    # Create circle actor
                    circle_mapper = vtk.vtkPolyDataMapper()
                    circle_mapper.SetInputConnection(circle_source.GetOutputPort())
                    circle_actor = vtk.vtkActor()
                    circle_actor.SetMapper(circle_mapper)
                    circle_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue for moment
                    circle_actor.GetProperty().SetLineWidth(2)
                    circle_actor.GetProperty().SetRepresentationToWireframe()
                    force_arrow_actors.append(circle_actor)
                    
                    # Add a small arrow on the circle to indicate direction
                    arrow_angle = 45 if force_vector[2] > 0 else 225
                    arrow_x = node.x + radius * math.cos(math.radians(arrow_angle))
                    arrow_y = node.y + radius * math.sin(math.radians(arrow_angle))
                    
                    arrow_source = vtk.vtkArrowSource()
                    arrow_source.SetTipResolution(16)
                    
                    arrow_transform = vtk.vtkTransform()
                    arrow_transform.Translate(arrow_x, arrow_y, 0)
                    arrow_transform.RotateZ(arrow_angle + (90 if force_vector[2] > 0 else -90))
                    arrow_transform.Scale(radius * 0.5, radius * 0.5, radius * 0.5)
                    
                    arrow_mapper = vtk.vtkPolyDataMapper()
                    arrow_mapper.SetInputConnection(arrow_source.GetOutputPort())
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(arrow_mapper)
                    arrow_actor.SetUserTransform(arrow_transform)
                    arrow_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue for moment
                    force_arrow_actors.append(arrow_actor)

    # --- 4. Create Actor for Deformed Shape (if data is available) ---
    deformed_actor = None
    if loadcombo and scaling:
        displace_scale = scaling.get("displacement", 100)
        deformed_points = vtk.vtkPoints()
        for node in nodes:
            deformed_points.InsertNextPoint(
                node.x_displaced(loadcombo, displace_scale),
                node.y_displaced(loadcombo, displace_scale),
                0
            )
        
        deformed_poly_data = vtk.vtkPolyData()
        deformed_poly_data.SetPoints(deformed_points)
        deformed_poly_data.SetLines(lines)

        deformed_mapper = vtk.vtkPolyDataMapper()
        deformed_mapper.SetInputData(deformed_poly_data)
        deformed_actor = vtk.vtkActor()
        deformed_actor.SetMapper(deformed_mapper)
        deformed_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        deformed_actor.GetProperty().SetLineStipplePattern(0xF0F0)
        deformed_actor.GetProperty().SetLineWidth(2)

    # --- 5. Set up the rendering pipeline ---
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor2D(node_label_actor)
    for label_actor in member_label_actors:
        renderer.AddActor(label_actor)
    for hinge_actor in hinge_actors:
        renderer.AddActor(hinge_actor)
    for force_actor in force_arrow_actors:
        renderer.AddActor(force_actor)
    if deformed_actor:
        renderer.AddActor(deformed_actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    # --- 6. Define Interactor and Keyboard Callbacks ---
    class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self, parent=None):
            self.parent = vtk.vtkRenderWindowInteractor()
            if parent is not None:
                self.parent = parent
            
            self.deformed_actor = deformed_actor
            self.node_label_actor = node_label_actor
            self.member_label_actors = member_label_actors
            self.hinge_actors = hinge_actors
            self.force_arrow_actors = force_arrow_actors
            self.AddObserver("KeyPressEvent", self.key_press_event)

        def key_press_event(self, obj, event):
            key = self.parent.GetKeySym()
            if key == 'd':
                if self.deformed_actor:
                    is_visible = self.deformed_actor.GetVisibility()
                    self.deformed_actor.SetVisibility(not is_visible)
            elif key == 'l':
                is_visible = self.node_label_actor.GetVisibility()
                self.node_label_actor.SetVisibility(not is_visible)
            elif key == 'm':
                if self.member_label_actors:
                    is_visible = self.member_label_actors[0].GetVisibility()
                    for act in self.member_label_actors:
                        act.SetVisibility(not is_visible)
            elif key == 'h':
                if self.hinge_actors:
                    is_visible = self.hinge_actors[0].GetVisibility()
                    for act in self.hinge_actors:
                        act.SetVisibility(not is_visible)
            elif key == 'f':
                if self.force_arrow_actors:
                    is_visible = self.force_arrow_actors[0].GetVisibility()
                    for act in self.force_arrow_actors:
                        act.SetVisibility(not is_visible)
                print("Press 'f' to toggle force arrows.")
            
            self.parent.GetRenderWindow().Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(KeyPressInteractorStyle(parent=interactor))

    # --- 7. Start the visualization ---
    render_window.Render()
    print("\n--- VTK Interaction ---")
    print("Press 'd' to toggle deformed shape.")
    print("Press 'l' to toggle node labels.")
    print("Press 'm' to toggle member labels.")
    print("Press 'h' to toggle hinge visualizations.")
    if force_arrow_actors:
        print("Press 'f' to toggle force arrows.")
    print("-----------------------\n")
    interactor.Start()

def plot_structure_loadcombos_vtk(nodes, members, loadcombos=None, scaling=None):
    """
    Visualizes the structure with results from multiple load combinations.
    
    Parameters
    ----------
    nodes : list
        A list of R2Node objects.
    members : list
        A list of R2Truss or R2Frame objects.
    loadcombos : list or single LoadCombo, optional
        The load combinations to display.
    scaling : dict, optional
        Scaling factors for deformations.
    """
    import vtk
    import math
    
    # Convert single loadcombo to list for consistent handling
    if loadcombos and not isinstance(loadcombos, list):
        loadcombos = [loadcombos]
        
    # --- Create base geometry (same for all load combinations) ---
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    line_colors = vtk.vtkUnsignedCharArray()
    line_colors.SetNumberOfComponents(3)
    line_colors.SetName("Colors")
    
    type_color_map = {"FRAME": (0, 0, 255), "TRUSS": (0, 255, 0)}
    default_color = (0, 0, 0)
    
    node_uid_to_vtk_id = {node.uid: i for i, node in enumerate(nodes)}
    for node in nodes:
        points.InsertNextPoint(node.x, node.y, 0)
    
    for member in members:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, node_uid_to_vtk_id[member.inode.uid])
        line.GetPointIds().SetId(1, node_uid_to_vtk_id[member.jnode.uid])
        lines.InsertNextCell(line)
        color = type_color_map.get(member.type, default_color)
        line_colors.InsertNextTuple3(color[0], color[1], color[2])
    
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    poly_data.GetCellData().SetScalars(line_colors)
    
    # --- Create deformed actors for each load combination ---
    deformed_actors = {}
    if loadcombos and scaling:
        displace_scale = scaling.get("displacement", 100)
        
        for combo in loadcombos:
            deformed_points = vtk.vtkPoints()
            for node in nodes:
                if combo.name in node.displacements:
                    deformed_points.InsertNextPoint(
                        node.x_displaced(combo, displace_scale),
                        node.y_displaced(combo, displace_scale),
                        0
                    )
                else:
                    # If no displacement data for this combo, use original position
                    deformed_points.InsertNextPoint(node.x, node.y, 0)
            
            deformed_poly_data = vtk.vtkPolyData()
            deformed_poly_data.SetPoints(deformed_points)
            deformed_poly_data.SetLines(lines)
            
            deformed_mapper = vtk.vtkPolyDataMapper()
            deformed_mapper.SetInputData(deformed_poly_data)
            deformed_actor = vtk.vtkActor()
            deformed_actor.SetMapper(deformed_mapper)
            deformed_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            deformed_actor.GetProperty().SetLineStipplePattern(0xF0F0)
            deformed_actor.GetProperty().SetLineWidth(2)
            deformed_actor.SetVisibility(False)  # Initially hidden
            
            deformed_actors[combo.name] = deformed_actor
    
    # --- Create renderer and add all actors ---
    renderer = vtk.vtkRenderer()
    
    # Add base geometry
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(3)
    renderer.AddActor(actor)
    
    # Add all deformed actors
    for actor in deformed_actors.values():
        renderer.AddActor(actor)
    
    # --- Add combo selection interface ---
    # Create combo selector text actors
    combo_text_actors = {}
    if loadcombos:
        for i, combo in enumerate(loadcombos):
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"[{i+1}] {combo.name}")
            text_actor.GetTextProperty().SetColor(0.2, 0.2, 0.8)
            text_actor.GetTextProperty().SetFontSize(14)
            text_actor.SetPosition(10, 10 + i*20)
            combo_text_actors[combo.name] = text_actor
            renderer.AddActor2D(text_actor)
    
    # Create "active combo" indicator
    active_combo_actor = vtk.vtkTextActor()
    active_combo_actor.SetInput("No active combination")
    active_combo_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
    active_combo_actor.GetTextProperty().SetFontSize(16)
    active_combo_actor.GetTextProperty().SetBold(True)
    active_combo_actor.SetPosition(10, 10 + len(loadcombos)*20 + 10)
    renderer.AddActor2D(active_combo_actor)
    
    # --- Set up the rendering window ---
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    # --- Define Interactor with keyboard controls for combo selection ---
    class MultiComboInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        def __init__(self, parent=None):
            self.parent = vtk.vtkRenderWindowInteractor()
            if parent is not None:
                self.parent = parent
            
            self.deformed_actors = deformed_actors
            self.loadcombos = loadcombos
            self.active_combo = None
            self.active_combo_actor = active_combo_actor
            self.AddObserver("KeyPressEvent", self.key_press_event)
        
        def key_press_event(self, obj, event):
            key = self.parent.GetKeySym()
            
            # Handle numeric keys for combo selection
            if key.isdigit() and int(key) > 0 and int(key) <= len(self.loadcombos):
                combo_idx = int(key) - 1
                selected_combo = self.loadcombos[combo_idx]
                
                # Hide all deformed actors
                for actor in self.deformed_actors.values():
                    actor.SetVisibility(False)
                
                # Show only the selected one
                if selected_combo.name in self.deformed_actors:
                    self.deformed_actors[selected_combo.name].SetVisibility(True)
                    self.active_combo = selected_combo
                    self.active_combo_actor.SetInput(f"Active: {selected_combo.name}")
            
            elif key == 'h':
                # Toggle help text
                pass  # Add help text toggle functionality
            
            self.parent.GetRenderWindow().Render()
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(MultiComboInteractorStyle(parent=interactor))
    
    # --- Start visualization ---
    render_window.Render()
    print("\n--- VTK Interaction ---")
    print("Press keys 1-{} to select different load combinations:".format(len(loadcombos) if loadcombos else 0))
    for i, combo in enumerate(loadcombos or []):
        print(f"  [{i+1}] {combo.name}")
    print("-----------------------\n")
    interactor.Start()