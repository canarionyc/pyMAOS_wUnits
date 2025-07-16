#!/usr/bin/env python
# matrix_heatmap_vtk.py - Visualize a matrix as a heatmap using VTK

import numpy as np
import vtk
import os
import sys

class MatrixInteractorStyle(vtk.vtkInteractorStyleImage):
    """Custom interactor style to handle clicks on the matrix"""
    
    def __init__(self, matrix, info_text, renderer):
        """Initialize with matrix data, info text actor, and renderer"""
        self.matrix = matrix
        self.height, self.width = matrix.shape
        self.info_text = info_text
        self.renderer = renderer  # Store the renderer directly
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
    
    def left_button_press_event(self, obj, event):
        """Handle left button press to show matrix position and value"""
        try:
            # Get click position
            click_pos = self.GetInteractor().GetEventPosition()
            print(f"Click position: {click_pos}")
            
            # Use stored renderer instead of GetCurrentRenderer()
            if not self.renderer:
                self.info_text.SetInput("Renderer not available")
                self.OnLeftButtonDown()
                return
                
            # Use the display to world coordinate transformation
            coordinate = vtk.vtkCoordinate()
            coordinate.SetCoordinateSystemToDisplay()
            coordinate.SetValue(click_pos[0], click_pos[1], 0)
            world_pos = coordinate.GetComputedWorldValue(self.renderer)
            
            print(f"World position: {world_pos}")
            
            # Check if the click is within the matrix bounds
            if (0 <= world_pos[0] <= self.width and 0 <= world_pos[1] <= self.height):
                # Convert to matrix indices
                x = int(min(self.width - 1, max(0, world_pos[0])))
                y = int(min(self.height - 1, max(0, self.height - world_pos[1])))
                
                # Get the value
                value = self.matrix[y, x]
                info_text = f"Position: [{x},{y}], Value: {value:.4f}"
                print(info_text)
                self.info_text.SetInput(info_text)
            else:
                self.info_text.SetInput(f"Click outside matrix area")
                
        except Exception as e:
            # Handle any other errors
            print(f"Error in click handling: {e}")
            self.info_text.SetInput(f"Error: {str(e)}")
            
        # Call the parent handler for normal interaction
        self.OnLeftButtonDown()

def visualize_matrix_heatmap(matrix_file, title="Matrix Heatmap", use_log_scale=False):
    """
    Visualize a matrix as a heatmap using VTK
    
    Parameters:
    -----------
    matrix_file : str
        Path to CSV file containing the matrix
    title : str
        Title for the visualization window
    use_log_scale : bool
        If True, use logarithmic scale for color mapping
    """
    # Load the matrix data
    matrix = np.loadtxt(matrix_file, delimiter=',')
    height, width = matrix.shape
    
    print(f"Loaded matrix with shape: {matrix.shape}")
    print(f"Value range: {matrix.min()} to {matrix.max()}")
    
    # Create points and cells for sharp boundaries
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    
    # Create grid of points - flip y axis (origin at bottom left)
    for y in range(height+1):
        for x in range(width+1):
            # Use height-y to flip the y-axis
            points.InsertNextPoint(x, height-y, 0)
    
    # Create cells as quads
    cellId = 0
    for y in range(height):
        for x in range(width):
            # Get the matrix value
            value = matrix[y, x]
            
            # Apply log scaling if requested
            if use_log_scale and value != 0:
                value = np.sign(value) * np.log10(abs(value) + 1)
                
            # Define the quad using point ids - adjusted for flipped y
            quad = vtk.vtkQuad()
            # Points order: bottom-left, bottom-right, top-right, top-left (for flipped y)
            quad.GetPointIds().SetId(0, y * (width+1) + x)
            quad.GetPointIds().SetId(1, y * (width+1) + x + 1)
            quad.GetPointIds().SetId(2, (y+1) * (width+1) + x + 1)
            quad.GetPointIds().SetId(3, (y+1) * (width+1) + x)
            
            cells.InsertNextCell(quad)
            scalars.InsertNextValue(value)
            cellId += 1
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.GetCellData().SetScalars(scalars)
    
    # Create a lookup table for color mapping
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.667, 0.0)  # Blue to red
    lut.Build()
    
    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetScalarModeToUseCellData()  # Use cell data for coloring
    
    # Set color mapping range
    if use_log_scale:
        max_val = np.sign(matrix.max()) * np.log10(abs(matrix.max()) + 1)
        min_val = np.sign(matrix.min()) * np.log10(abs(matrix.min()) + 1)
    else:
        max_val = matrix.max()
        min_val = matrix.min()
        
    mapper.SetScalarRange(min_val, max_val)
    
    # Create cell actor
    cell_actor = vtk.vtkActor()
    cell_actor.SetMapper(mapper)
    
    # Add grid lines for better visibility
    grid_actor = vtk.vtkActor()
    grid_mapper = vtk.vtkPolyDataMapper()
    
    # Create grid lines
    grid_lines = vtk.vtkAppendPolyData()
    
    # Create horizontal and vertical grid lines
    for i in range(height+1):
        line = vtk.vtkLineSource()
        line.SetPoint1(0, height-i, 0)
        line.SetPoint2(width, height-i, 0)
        line.Update()
        grid_lines.AddInputData(line.GetOutput())
    
    for i in range(width+1):
        line = vtk.vtkLineSource()
        line.SetPoint1(i, 0, 0)
        line.SetPoint2(i, height, 0)
        line.Update()
        grid_lines.AddInputData(line.GetOutput())
    
    grid_lines.Update()
    grid_mapper.SetInputData(grid_lines.GetOutput())
    grid_actor.SetMapper(grid_mapper)
    grid_actor.GetProperty().SetColor(0.3, 0.3, 0.3)  # Dark grey grid lines
    grid_actor.GetProperty().SetLineWidth(1.0)
    
    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    render_window.SetWindowName(title)
    
    # Add actors to renderer
    renderer.AddActor(cell_actor)
    renderer.AddActor(grid_actor)
    renderer.SetBackground(0.2, 0.2, 0.2)
    
    # Add cell value labels
    if width <= 20 and height <= 20:  # Only show labels for small matrices
        for y in range(height):
            for x in range(width):
                value = matrix[y, x]
                
                # Create text actor
                text_actor = vtk.vtkTextActor()
                text_actor.SetInput(f"{value:.1f}")
                text_actor.GetTextProperty().SetFontSize(8)
                text_actor.GetTextProperty().SetColor(0.9, 0.9, 0.9)
                
                # Position in the center of the cell - adjusted for flipped y
                text_actor.SetPosition(x + 0.5, height - y - 0.5)
                renderer.AddActor(text_actor)
    
    # Add a color bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(mapper.GetLookupTable())
    scalar_bar.SetTitle("Matrix Value")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetPosition(0.9, 0.1)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    renderer.AddViewProp(scalar_bar)  # Updated from AddActor2D
    
    # Create text actor for displaying click information
    info_text = vtk.vtkTextActor()
    info_text.SetInput("Click on matrix to see values")
    info_text.GetTextProperty().SetFontSize(14)
    info_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    info_text.GetTextProperty().SetBackgroundColor(0.1, 0.1, 0.1)
    info_text.GetTextProperty().SetBackgroundOpacity(0.7)
    info_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    info_text.SetPosition(0.5, 0.95)  # Top center of window, away from the matrix
    info_text.GetTextProperty().SetJustificationToCentered()  # Center the text horizontally
    renderer.AddViewProp(info_text)  # Updated from AddActor2D
    
    # Setup camera to view the entire matrix
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(height * 0.6)  # Adjust to show entire matrix
    camera.SetPosition(width/2, height/2, 10)
    camera.SetFocalPoint(width/2, height/2, 0)
    
    # Create an interactor with custom image style
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Use custom style that handles clicks and shows values
    style = MatrixInteractorStyle(matrix, info_text, renderer)
    interactor.SetInteractorStyle(style)

    # Initialize and start
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    if False and len(sys.argv) > 1:
        matrix_file = sys.argv[1]
    else:
        matrix_file = "example_4_2_output/KSTRUCT.csv"
    
    if not os.path.exists(matrix_file):
        print(f"Error: File {matrix_file} not found")
        sys.exit(1)
        
    use_log_scale = "--log" in sys.argv
    try:
        visualize_matrix_heatmap(matrix_file, 
                                title=f"Matrix Heatmap: {os.path.basename(matrix_file)}", 
                                use_log_scale=use_log_scale)
    except Exception as e:
        print(f"Error visualizing matrix: {e}")
