def plot_loadcombos_vtk(self, loadcombos=None, scaling=None):
    """Visualizes the structure with results from multiple load combinations using VTK."""
    try:
        # Import the plotting function from R2Structure_extras
        from pyMAOS.R2Structure_extras import plot_loadcombos_vtk as plot_vtk_impl
        from pyMAOS.R2Structure_extras import check_vtk_available
        
        # Check if VTK is available
        if not check_vtk_available():
            print("Warning: VTK library is not installed. Please install VTK for visualization.")
            return
            
        # Call the imported function with self as first argument
        plot_vtk_impl(self, loadcombos, scaling)
    except ImportError as e:
        print(f"Warning: Visualization module not found: {e}")
        print("Make sure R2Structure_extras.py is in the pyMAOS package directory.")
    except Exception as e:
        print(f"Error during visualization: {e}")
