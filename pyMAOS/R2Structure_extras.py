# -*- coding: utf-8 -*-
"""
Additional functionality for the R2Structure class, focused on visualization.
This module contains plotting and visualization methods that extend the core 
structural analysis capabilities.
"""
def check_vtk_available():
    """Check if VTK is available for visualization"""
    try:
        import vtk
        return True
    except ImportError:
        return False

def plot_loadcombos_vtk(structure, loadcombos=None, scaling=None):
    """
    Visualizes the structure with results from multiple load combinations using VTK.

    Parameters
    ----------
    structure : R2Structure
        The structural model to visualize
    loadcombos : list or single LoadCombo, optional
        The load combinations to display.
    scaling : dict, optional
        Scaling factors for deformations.
    """
    import vtk
    
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

    # Use the class's nodes and members
    node_uid_to_vtk_id = {node.uid: i for i, node in enumerate(structure.nodes)}
    for node in structure.nodes:
        points.InsertNextPoint(node.x, node.y, 0)

    for member in structure.members:
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
            for node in structure.nodes:
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

    # --- Set up the rendering window ---
    renderer.SetBackground(1, 1, 1)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

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
                self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_event)
    
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
    if loadcombos:
        print("\n--- VTK Interaction ---")
        print("Press keys 1-{} to select different load combinations:".format(len(loadcombos) if loadcombos else 0))
        for i, combo in enumerate(loadcombos or []):
            print(f"  [{i+1}] {combo.name}")
        print("-----------------------\n")
        interactor.Start()
