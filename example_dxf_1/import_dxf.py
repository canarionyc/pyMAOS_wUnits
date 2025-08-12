import ezdxf
import os
import numpy as np
from pyMAOS.pymaos_sections import Section

def import_2d_model_from_dxf(dxf_file, default_section=None, tolerance=1e-6):
    """
    Import a 2D structural model from a DXF file.

    Parameters
    ----------
    dxf_file : str
        Path to the DXF file
    default_section : Section, optional
        Default section to assign to all members
    tolerance : float, optional
        Tolerance for node merging

    Returns
    -------
    dict
        Dictionary containing 'nodes' and 'members'
    """
    if not os.path.exists(dxf_file):
        raise FileNotFoundError(f"DXF file not found: {dxf_file}")

    print(f"Opening DXF file: {dxf_file}")
    doc = ezdxf.readfile(dxf_file)

    # Get the modelspace
    msp = doc.modelspace()

    # Extract nodes from LINE entities
    nodes = []
    members = []
    node_map = {}  # Maps coordinates to node indices

    print("Extracting structural elements...")

    # Process LINE entities as structural members
    for entity in msp.query('LINE'):
        start_point = entity.dxf.start
        end_point = entity.dxf.end

        # Convert to tuples for hashing
        start_tuple = (round(start_point.x, 9), round(start_point.y, 9))
        end_tuple = (round(end_point.x, 9), round(end_point.y, 9))

        # Create or get start node index
        if start_tuple not in node_map:
            node_map[start_tuple] = len(nodes)
            nodes.append({"id": len(nodes), "x": start_point.x, "y": start_point.y, "z": 0.0})
        start_node_id = node_map[start_tuple]

        # Create or get end node index
        if end_tuple not in node_map:
            node_map[end_tuple] = len(nodes)
            nodes.append({"id": len(nodes), "x": end_point.x, "y": end_point.y, "z": 0.0})
        end_node_id = node_map[end_tuple]

        # Skip zero-length members
        if start_node_id != end_node_id:
            # Extract any custom properties from the entity
            layer = entity.dxf.layer

            # Create the member
            members.append({
                "id": len(members),
                "start_node": start_node_id,
                "end_node": end_node_id,
                "section": default_section,
                "layer": layer
            })

    print(f"Found {len(nodes)} nodes and {len(members)} members")

    # Check if any TEXT entities exist that might contain properties
    for text in msp.query('TEXT'):
        print(f"Found text: {text.dxf.text} at position ({text.dxf.insert.x}, {text.dxf.insert.y})")
        # Could be used for annotations, section assignments, etc.

    return {
        "nodes": nodes,
        "members": members
    }

def calculate_member_lengths(model):
    """Calculate the length of each member in the model"""
    for member in model["members"]:
        start_node = model["nodes"][member["start_node"]]
        end_node = model["nodes"][member["end_node"]]

        dx = end_node["x"] - start_node["x"]
        dy = end_node["y"] - start_node["y"]
        dz = end_node["z"] - start_node["z"]

        length = np.sqrt(dx**2 + dy**2 + dz**2)
        member["length"] = length

    return model

# Example usage:
if __name__ == "__main__":
    # Create a default section
    default_section = Section(uid=1, Area="10 in^2", Ixx="100 in^4", Iyy="50 in^4", name="Default")

    # Import the model
    dxf_path = os.path.join(os.getenv('PAYMAOS_HOME', 'C:\\dev\\pyMAOS_testing\\'), "example_dxf_1", "cantilever_beam.dxf")
    print(dxf_path)

    os.path.exists(dxf_path)
    model = import_2d_model_from_dxf(dxf_path, default_section=default_section)
    print(model)
    # Calculate member lengths
    model = calculate_member_lengths(model)

    # Print model summary
    print("\nModel Summary:")
    print(f"Total nodes: {len(model['nodes'])}")
    print(f"Total members: {len(model['members'])}")

    # Print first few nodes and members
    print("\nSample Nodes:")
    for i, node in enumerate(model["nodes"][:3]):
        print(f"Node {i}: ({node['x']:.3f}, {node['y']:.3f}, {node['z']:.3f})")

    print("\nSample Members:")
    for i, member in enumerate(model["members"][:3]):
        print(f"Member {i}: Nodes {member['start_node']}-{member['end_node']}, Length: {member['length']:.3f}")