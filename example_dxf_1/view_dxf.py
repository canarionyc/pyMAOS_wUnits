import os
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

def view_dxf(dxf_path):
    """
    Plot a DXF file directly using ezdxf and matplotlib
    """
    # Load the DXF file
    doc = ezdxf.readfile(dxf_path)

    # Initialize matplotlib figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    # Create the rendering context
    ctx = RenderContext(doc)

    # Create matplotlib backend
    backend = MatplotlibBackend(ax)

    # Create frontend and draw all entities
    Frontend(ctx, backend).draw_layout(doc.modelspace())

    # Add some labels
    plt.title("DXF Viewer: " + dxf_path)

    # Adjust the plot
    ax.set_aspect('equal')
    ax.autoscale(True)
    ax.margins(0.1)

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    dxf_file = "cantilever_beam.dxf"
    os.path.exists(dxf_file)
    view_dxf(dxf_file)