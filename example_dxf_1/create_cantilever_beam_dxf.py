import ezdxf
import os

def create_cantilever_beam_dxf(filename, length=10, height=0):
    """
    Create a DXF file with a simple cantilever beam.

    Parameters:
    -----------
    filename : str
        Output DXF filename
    length : float
        Length of the beam
    height : float
        Height position of the beam
    """
    # Create a new DXF document (using the latest DXF version by default)
    doc = ezdxf.new()

    # Add a new modelspace
    msp = doc.modelspace()

    # Create a horizontal line representing the cantilever beam
    # Fixed at (0,0) and extending to (length,0)
    msp.add_line((0, height, 0), (length, height, 0), dxfattribs={"layer": "BEAM"})

    # Add a small vertical line at the fixed end to indicate the support
    support_size = length * 0.1
    msp.add_line((0, height - support_size, 0), (0, height + support_size, 0),
                dxfattribs={"layer": "SUPPORT"})

    # Create examples directory if it doesn't exist
    examples_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        print(f"Created directory: {examples_dir}")

    # Save the DXF file
    filepath = os.path.join(examples_dir, filename)
    doc.saveas(filepath)
    print(f"Cantilever beam DXF created at: {filepath}")
    return filepath

if __name__ == "__main__":
    # Create a cantilever beam DXF file with a 10-unit length beam
    create_cantilever_beam_dxf("simple_frame.dxf", length=10)