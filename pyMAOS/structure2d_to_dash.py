import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go

def generate_dash_report(structure, loadcombos, output_file="interactive_report.html"):
    """
    Generate an interactive HTML report using Plotly.

    Parameters
    ----------
    structure : R2Structure
        The structure object containing nodes, members, and results.
    loadcombos : list
        List of load combinations to include in the report.
    output_file : str
        Path to save the HTML report.

    Returns
    -------
    str
        Path to the created HTML file
    """
    print(f"Generating interactive HTML report: {output_file}")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    # Extract values safely (handling Pint quantities)
    def extract_value(val):
        try:
            if hasattr(val, 'magnitude'):  # Check if it's a Pint quantity
                return float(val.magnitude)
            return float(val)
        except Exception as e:
            print(f"Warning: Error extracting value: {e}")
            return 0.0

    # Create a plotly figure with correct subplot types
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}],  # Changed from "scatter3d" to default (xy)
               [{}, {"type": "table"}]],
        subplot_titles=("Structure Visualization", "Deformed Shape",
                        "Bending Moment Diagram", "Results Summary")
    )

    # Plot 1: Structure Geometry (same code, now working with xy subplot)
    print("Creating structure geometry plot...")
    # Add members
    for member in structure.members:
        fig.add_trace(
            go.Scatter(
                x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                mode='lines',
                line=dict(color='blue', width=3),
                name=f"Member {member.uid}"
            ),
            row=1, col=1
        )

    # Rest of the code remains the same...

    # Update layout
    fig.update_layout(
        title_text=f"Structural Analysis Results - {loadcombos[0].name if hasattr(loadcombos[0], 'name') else 'Default'}",
        height=900,
        width=1200,
        showlegend=False
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write the HTML file directly using Plotly
    print(f"Writing HTML report to: {output_file}")
    try:
        fig.write_html(output_file, full_html=True, include_plotlyjs='cdn')
        print(f"Successfully created HTML report at: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return None