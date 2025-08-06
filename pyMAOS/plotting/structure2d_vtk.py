# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.lines import Line2D



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
    points = vtk.vtkPoints(); points.debug = True
    lines = vtk.vtkCellArray(); lines.debug = True
    line_colors = vtk.vtkUnsignedCharArray()
    line_colors.SetNumberOfComponents(3)
    line_colors.SetName("Colors")

    # Update the color mapping to use black for everything
    type_color_map = {"FRAME": (0, 0, 0), "TRUSS": (0, 0, 0)}
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
    # Later in the code, update node label colors to black
    node_label_mapper.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)  # Black text
    node_label_mapper.GetLabelTextProperty().SetFontSize(12)  # Set specific font size
    node_label_actor = vtk.vtkActor2D()
    node_label_actor.SetMapper(node_label_mapper)

    # --- Create Member Labels with same approach as node labels ---
    member_labels_points = vtk.vtkPoints()
    member_labels_poly = vtk.vtkPolyData()
    member_labels = vtk.vtkStringArray()
    member_labels.SetName("MemberLabels")
    
    for member in members:
        # Calculate midpoint for label placement
        mid_x = (member.inode.x + member.jnode.x) / 2
        mid_y = (member.inode.y + member.jnode.y) / 2
        member_labels_points.InsertNextPoint(mid_x, mid_y, 0)
        member_labels.InsertNextValue(f"M{member.uid}")
    
    member_labels_poly.SetPoints(member_labels_points)
    member_labels_poly.GetPointData().AddArray(member_labels)
    
    member_label_mapper = vtk.vtkLabeledDataMapper()
    member_label_mapper.SetInputData(member_labels_poly)
    member_label_mapper.SetFieldDataName("MemberLabels")
    member_label_mapper.SetLabelModeToLabelFieldData()
    # Member label colors to black
    member_label_mapper.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)  # Black text
    member_label_mapper.GetLabelTextProperty().SetFontSize(12)  # Same font size as nodes
    
    member_label_actor = vtk.vtkActor2D()
    member_label_actor.SetMapper(member_label_mapper)

    # --- 3. Create direction arrows without the duplicate member labels ---
    # We no longer need member_label_actors since we have member_label_actor
    member_label_actors = []  # Keep this empty list for compatibility with existing code

    # --- 3c. Create direction arrows to show member orientation ---
    direction_arrow_actors = []
    for member in members:
        p1 = np.array([member.inode.x, member.inode.y, 0])
        p2 = np.array([member.jnode.x, member.jnode.y, 0])
        
        # Calculate midpoint (slightly offset from true midpoint to avoid overlapping with label)
        mid_point = p1 + 0.6 * (p2 - p1)  # Position at 60% along member
        
        # Get direction vector and normalize it
        dir_vector = p2 - p1
        length = np.linalg.norm(dir_vector)
        if length > 0:
            dir_vector = dir_vector / length
        
        # Create arrow source
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipResolution(12)
        arrow_source.SetShaftResolution(12)
        
        # Scale and position the arrow
        arrow_length = min(3.0, length * 0.2)  # Arrow length proportional to member but capped
        
        # Calculate rotation angle in degrees
        angle_rad = math.atan2(dir_vector[1], dir_vector[0])
        angle_deg = math.degrees(angle_rad)
        
        # Create transform for arrow
        arrow_transform = vtk.vtkTransform()
        arrow_transform.Translate(mid_point)
        arrow_transform.RotateZ(angle_deg)
        arrow_transform.Scale(arrow_length, arrow_length, arrow_length)
        
        # Create arrow actor
        arrow_mapper = vtk.vtkPolyDataMapper()
        arrow_mapper.SetInputConnection(arrow_source.GetOutputPort())
        arrow_actor = vtk.vtkActor()
        arrow_actor.SetMapper(arrow_mapper)
        arrow_actor.SetUserTransform(arrow_transform)
        # Direction arrows to black
        arrow_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black arrows
        direction_arrow_actors.append(arrow_actor)

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
                # Hinge actors to black
                hinge_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black hinges
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
                # Hinge actors to black
                hinge_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black hinges
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
                    # Force arrows to black (in the node loads section)
                    # For X-direction force arrows
                    arrow_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black for X-force
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
                    # For Y-direction force arrows
                    arrow_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black for Y-force
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
                    # For moment visualization
                    circle_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black for moment circle
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
                    # For moment visualization
                    arrow_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black for moment arrow
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
        # Deformed shape to gray (keep this slightly different for visibility)
        deformed_actor.GetProperty().SetColor(0.5, 0.5, 0.5)  # Gray for deformed shape
        deformed_actor.GetProperty().SetLineStipplePattern(0xF0F0)
        deformed_actor.GetProperty().SetLineWidth(2)

    # --- 5. Set up the rendering pipeline ---
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor2D(node_label_actor)
    renderer.AddActor2D(member_label_actor)  # Add the new member label actor

    for hinge_actor in hinge_actors:
        renderer.AddActor(hinge_actor)
    for force_actor in force_arrow_actors:
        renderer.AddActor(force_actor)
    for direction_arrow_actor in direction_arrow_actors:
        renderer.AddActor(direction_arrow_actor)
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
            self.member_label_actor = member_label_actor  # Add reference to member labels
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
                is_visible = self.member_label_actor.GetVisibility()
                self.member_label_actor.SetVisibility(not is_visible)
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

# def plot_structure_loadcombos_vtk(nodes, members, loadcombos=None, scaling=None):
#     """
#     Visualizes the structure with results from multiple load combinations.
    
#     Parameters
#     ----------
#     nodes : list
#         A list of R2Node objects.
#     members : list
#         A list of R2Truss or R2Frame objects.
#     loadcombos : list or single LoadCombo, optional
#         The load combinations to display.
#     scaling : dict, optional
#         Scaling factors for deformations.
#     """
#     import vtk
#     import math
    
#     # Convert single loadcombo to list for consistent handling
#     if loadcombos and not isinstance(loadcombos, list):
#         loadcombos = [loadcombos]
        
#     # --- Create base geometry (same for all load combinations) ---
#     points = vtk.vtkPoints()
#     lines = vtk.vtkCellArray()
#     line_colors = vtk.vtkUnsignedCharArray()
#     line_colors.SetNumberOfComponents(3)
#     line_colors.SetName("Colors")
    
#     type_color_map = {"FRAME": (0, 0, 255), "TRUSS": (0, 255, 0)}
#     default_color = (0, 0, 0)
    
#     node_uid_to_vtk_id = {node.uid: i for i, node in enumerate(nodes)}
#     for node in nodes:
#         points.InsertNextPoint(node.x, node.y, 0)
    
#     for member in members:
#         line = vtk.vtkLine()
#         line.GetPointIds().SetId(0, node_uid_to_vtk_id[member.inode.uid])
#         line.GetPointIds().SetId(1, node_uid_to_vtk_id[member.jnode.uid])
#         lines.InsertNextCell(line)
#         color = type_color_map.get(member.type, default_color)
#         line_colors.InsertNextTuple3(color[0], color[1], color[2])
    
#     poly_data = vtk.vtkPolyData()
#     poly_data.SetPoints(points)
#     poly_data.SetLines(lines)
#     poly_data.GetCellData().SetScalars(line_colors)
    
#     # --- Create deformed actors for each load combination ---
#     deformed_actors = {}
#     if loadcombos and scaling:
#         displace_scale = scaling.get("displacement", 100)
        
#         for combo in loadcombos:
#             deformed_points = vtk.vtkPoints()
#             for node in nodes:
#                 if combo.name in node.displacements:
#                     deformed_points.InsertNextPoint(
#                         node.x_displaced(combo, displace_scale),
#                         node.y_displaced(combo, displace_scale),
#                         0
#                     )
#                 else:
#                     # If no displacement data for this combo, use original position
#                     deformed_points.InsertNextPoint(node.x, node.y, 0)
            
#             deformed_poly_data = vtk.vtkPolyData()
#             deformed_poly_data.SetPoints(deformed_points)
#             deformed_poly_data.SetLines(lines)
            
#             deformed_mapper = vtk.vtkPolyDataMapper()
#             deformed_mapper.SetInputData(deformed_poly_data)
#             deformed_actor = vtk.vtkActor()
#             deformed_actor.SetMapper(deformed_mapper)
#             deformed_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
#             deformed_actor.GetProperty().SetLineStipplePattern(0xF0F0)
#             deformed_actor.GetProperty().SetLineWidth(2)
#             deformed_actor.SetVisibility(False)  # Initially hidden
            
#             deformed_actors[combo.name] = deformed_actor
    
#     # --- Create renderer and add all actors ---
#     renderer = vtk.vtkRenderer()
    
#     # Add base geometry
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(poly_data)
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetLineWidth(3)
#     renderer.AddActor(actor)
    
#     # Add all deformed actors
#     for actor in deformed_actors.values():
#         renderer.AddActor(actor)
    
#     # --- Add combo selection interface ---
#     # Create combo selector text actors
#     combo_text_actors = {}
#     if loadcombos:
#         for i, combo in enumerate(loadcombos):
#             text_actor = vtk.vtkTextActor()
#             text_actor.SetInput(f"[{i+1}] {combo.name}")
#             text_actor.GetTextProperty().SetColor(0.2, 0.2, 0.8)
#             text_actor.GetTextProperty().SetFontSize(14)
#             text_actor.SetPosition(10, 10 + i*20)
#             combo_text_actors[combo.name] = text_actor
#             renderer.AddActor2D(text_actor)
    
#     # Create "active combo" indicator
#     active_combo_actor = vtk.vtkTextActor()
#     active_combo_actor.SetInput("No active combination")
#     active_combo_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
#     active_combo_actor.GetTextProperty().SetFontSize(16)
#     active_combo_actor.GetTextProperty().SetBold(True)
#     active_combo_actor.SetPosition(10, 10 + len(loadcombos)*20 + 10)
#     renderer.AddActor2D(active_combo_actor)
    
#     # --- Set up the rendering window ---
#     renderer.SetBackground(1, 1, 1)
#     render_window = vtk.vtkRenderWindow()
#     render_window.AddRenderer(renderer)
#     render_window.SetSize(800, 600)
    
#     # --- Define Interactor with keyboard controls for combo selection ---
#     class MultiComboInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
#         def __init__(self, parent=None):
#             self.parent = vtk.vtkRenderWindowInteractor()
#             if parent is not None:
#                 self.parent = parent
            
#             self.deformed_actors = deformed_actors
#             self.loadcombos = loadcombos
#             self.active_combo = None
#             self.active_combo_actor = active_combo_actor
#             self.AddObserver("KeyPressEvent", self.key_press_event)
        
#         def key_press_event(self, obj, event):
#             key = self.parent.GetKeySym()
            
#             # Handle numeric keys for combo selection
#             if key.isdigit() and int(key) > 0 and int(key) <= len(self.loadcombos):
#                 combo_idx = int(key) - 1
#                 selected_combo = self.loadcombos[combo_idx]
                
#                 # Hide all deformed actors
#                 for actor in self.deformed_actors.values():
#                     actor.SetVisibility(False)
                
#                 # Show only the selected one
#                 if selected_combo.name in self.deformed_actors:
#                     self.deformed_actors[selected_combo.name].SetVisibility(True)
#                     self.active_combo = selected_combo
#                     self.active_combo_actor.SetInput(f"Active: {selected_combo.name}")
            
#             elif key == 'h':
#                 # Toggle help text
#                 pass  # Add help text toggle functionality
            
#             self.parent.GetRenderWindow().Render()
    
#     interactor = vtk.vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(render_window)
#     interactor.SetInteractorStyle(MultiComboInteractorStyle(parent=interactor))
    
#     # --- Start visualization ---
#     render_window.Render()
#     print("\n--- VTK Interaction ---")
#     print("Press keys 1-{} to select different load combinations:".format(len(loadcombos) if loadcombos else 0))
#     for i, combo in enumerate(loadcombos or []):
#         print(f"  [{i+1}] {combo.name}")
#     print("-----------------------\n")
#     interactor.Start()