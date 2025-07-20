import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from pathlib import Path

# Create simple classes to represent nodes and members for plotting
class SimpleNode:
    def __init__(self, uid, x, y, restraints=None):
        self.uid = uid
        self.x = x
        self.y = y
        self.restraints = restraints or [0, 0, 0]

class SimpleMember:
    def __init__(self, uid, inode, jnode, member_type="FRAME"):
        self.uid = uid
        self.inode = inode
        self.jnode = jnode
        self.type = member_type

def plot_structure_matplotlib(nodes, members):
    """Simple version of the plot_structure_matplotlib function for JSON data"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes and their labels
    for node in nodes:
        ax.plot(node.x, node.y, marker="o", markersize=6, color="red", 
                label="Node" if "Node" not in ax.get_legend_handles_labels()[1] else "")
        
        # Dynamic label offset based on coordinate magnitude
        offset = max(5, abs(node.y) * 0.02) if node.y != 0 else 5
        ax.text(node.x, node.y + offset, f'N{node.uid}', 
                color='darkred', fontsize=9, ha='center')

        # Visualize restraints
        if hasattr(node, "restraints") and node.restraints:
            rx, ry, rz = node.restraints
            # Dynamic size based on coordinates
            size = max(5, abs(node.x) * 0.02, abs(node.y) * 0.02) if (node.x != 0 or node.y != 0) else 5
            
            if rx:  # Horizontal restraint
                ax.plot([node.x - size, node.x + size], [node.y, node.y], color="blue", linewidth=2, 
                        label="Restraint (Ux)" if "Restraint (Ux)" not in ax.get_legend_handles_labels()[1] else "")
            if ry:  # Vertical restraint
                ax.plot([node.x, node.x], [node.y - size, node.y + size], color="green", linewidth=2, 
                        label="Restraint (Uy)" if "Restraint (Uy)" not in ax.get_legend_handles_labels()[1] else "")
            if rz:  # Rotational restraint
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(node.x + (size / 2) * np.cos(theta), node.y + (size / 2) * np.sin(theta), 
                        color="purple", linewidth=1.5,
                        label="Restraint (Rz)" if "Restraint (Rz)" not in ax.get_legend_handles_labels()[1] else "")

    # Plot members and their labels
    for member in members:
        ax.plot([member.inode.x, member.jnode.x], [member.inode.y, member.jnode.y],
                color="black", linewidth=1.5,
                label="Element" if "Element" not in ax.get_legend_handles_labels()[1] else "")
        
        # Calculate midpoint for the label
        mid_x = (member.inode.x + member.jnode.x) / 2
        mid_y = (member.inode.y + member.jnode.y) / 2
        ax.text(mid_x, mid_y, f'M{member.uid}', color='darkblue', fontsize=9, 
                ha='center', va='center', backgroundcolor=(1,1,1,0.7))

    # Add grid, labels, and legend
    ax.grid(True)
    ax.set_aspect("equal", "box")
    ax.set_title("Structure Visualization", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend()

    return fig, ax

def json_to_excel(json_file, output_file=None):
    """Convert structural analysis JSON results to Excel format with multiple sheets"""
    
    # Set default output filename in the same path as input file
    if output_file is None:
        input_path = Path(json_file)
        output_file = input_path.parent / (input_path.stem + '.xlsx')
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract units for column headers
    units = data['units']
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Get the workbook and create formats
        workbook = writer.book
        
        # Create formats - using try/except to handle potential issues
        try:
            header_format = workbook.add_format({
                'bold': True, 
                'text_wrap': True, 
                'valign': 'top',
                'fg_color': '#D7E4BC', 
                'border': 1
            })
        except AttributeError:
            # Fallback if xlsxwriter methods are not available
            header_format = None
        
        # 1. Summary (formerly Analysis Info) - now first sheet
        analysis_df = pd.DataFrame.from_dict(data['analysis_info'], orient='index', columns=['Value'])
        analysis_df.index.name = 'Parameter'
        analysis_df.to_excel(writer, sheet_name='Summary')
        
        # 2. Structure Visualization - second sheet
        # Create simple objects for plotting
        simple_nodes = []
        node_objects = {}  # For reference in member creation
        for node_id, node in data['nodes'].items():
            simple_node = SimpleNode(
                uid=node_id,
                x=node['coordinates']['x'],
                y=node['coordinates']['y'],
                restraints=node['restraints']
            )
            simple_nodes.append(simple_node)
            node_objects[node_id] = simple_node
        
        simple_members = []
        for member_id, member in data['members'].items():
            i_node_id = str(member['connectivity']['i_node'])
            j_node_id = str(member['connectivity']['j_node'])
            
            simple_member = SimpleMember(
                uid=member_id,
                inode=node_objects[i_node_id],
                jnode=node_objects[j_node_id],
                member_type=member['properties']['type']
            )
            simple_members.append(simple_member)
        
        # Create the plot
        fig, ax = plot_structure_matplotlib(simple_nodes, simple_members)
        
        # Add visualization to Excel
        try:
            worksheet = workbook.add_worksheet('Structure Visualization')
            
            # Save the figure to a BytesIO object
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format='png', dpi=150, bbox_inches='tight')
            imgdata.seek(0)
            
            # Insert the image into the worksheet
            worksheet.insert_image('A1', 'structure.png', {'image_data': imgdata, 'x_scale': 0.8, 'y_scale': 0.8})
        except AttributeError:
            print("Warning: Could not add structure visualization to Excel file")
        
        # Close the matplotlib figure to free memory
        plt.close(fig)
        
        # 3. Units sheet
        units_df = pd.DataFrame.from_dict(data['units'], orient='index', columns=['Unit'])
        units_df.index.name = 'Dimension'
        units_df.to_excel(writer, sheet_name='Units')
        
        # 4. Node information sheet with units in headers
        nodes_data = []
        for node_id, node in data['nodes'].items():
            node_info = {
                'Node ID': node_id,
                f'X ({units["length"]})': node['coordinates']['x'],
                f'Y ({units["length"]})': node['coordinates']['y'],
                'Restrained X': node['restraints'][0],
                'Restrained Y': node['restraints'][1],
                'Restrained Z': node['restraints'][2],
            }
            
            # Add displacements
            disp = node['displacements']['D']
            node_info.update({
                f'Displacement X ({units["length"]})': disp['ux'],
                f'Displacement Y ({units["length"]})': disp['uy'],
                'Rotation Z (rad)': disp['rz'],
            })
            
            # Add reactions
            reaction = node['reactions']['D']
            node_info.update({
                f'Reaction X ({units["force"]})': reaction['rx'],
                f'Reaction Y ({units["force"]})': reaction['ry'],
                f'Moment Z ({units["moment"]})': reaction['mz'],
            })
            
            nodes_data.append(node_info)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
        
        # 5. Member properties with units
        members_data = []
        for member_id, member in data['members'].items():
            member_info = {
                'Member ID': member_id,
                'i-node': member['connectivity']['i_node'],
                'j-node': member['connectivity']['j_node'],
                f'Length ({units["length"]})': member['properties']['length'],
                'Type': member['properties']['type'],
                'Material ID': member['properties']['material']['id'],
                f'E ({units["pressure"]})': member['properties']['material']['E'],
                'Section ID': member['properties']['section']['id'],
                f'Area ({units["length"]}²)': member['properties']['section']['area'],
                f'Ixx ({units["length"]}⁴)': member['properties']['section']['Ixx'],
            }
            members_data.append(member_info)
        
        members_df = pd.DataFrame(members_data)
        members_df.to_excel(writer, sheet_name='Member Properties', index=False)
        
        # 6. Member forces - Global and Local with units
        global_forces = []
        for member_id, member in data['members'].items():
            # i-node forces
            i_node_global = member['forces']['D']['global']['i_node']
            global_forces.append({
                'Member ID': member_id,
                'Node': f"{member['connectivity']['i_node']} (i)",
                f'Fx ({units["force"]})': i_node_global['fx'],
                f'Fy ({units["force"]})': i_node_global['fy'],
                f'Mz ({units["moment"]})': i_node_global['mz'],
                'System': 'Global'
            })
            
            # j-node forces
            j_node_global = member['forces']['D']['global']['j_node']
            global_forces.append({
                'Member ID': member_id,
                'Node': f"{member['connectivity']['j_node']} (j)",
                f'Fx ({units["force"]})': j_node_global['fx'],
                f'Fy ({units["force"]})': j_node_global['fy'],
                f'Mz ({units["moment"]})': j_node_global['mz'],
                'System': 'Global'
            })
        
        # Member forces - Local
        local_forces = []
        for member_id, member in data['members'].items():
            # i-node forces
            i_node_local = member['forces']['D']['local']['i_node']
            local_forces.append({
                'Member ID': member_id,
                'Node': f"{member['connectivity']['i_node']} (i)",
                f'Fx ({units["force"]})': i_node_local['fx'],
                f'Fy ({units["force"]})': i_node_local['fy'],
                f'Mz ({units["moment"]})': i_node_local['mz'],
                'System': 'Local'
            })
            
            # j-node forces
            j_node_local = member['forces']['D']['local']['j_node']
            local_forces.append({
                'Member ID': member_id,
                'Node': f"{member['connectivity']['j_node']} (j)",
                f'Fx ({units["force"]})': j_node_local['fx'],
                f'Fy ({units["force"]})': j_node_local['fy'],
                f'Mz ({units["moment"]})': j_node_local['mz'],
                'System': 'Local'
            })
            
        # Combine global and local forces into one sheet with sections
        all_forces = pd.DataFrame(global_forces + local_forces)
        all_forces.to_excel(writer, sheet_name='Member Forces', index=False)
        
        # Apply formats to all sheets (except visualization) - only if header_format is available
        if header_format:
            for sheet_name in writer.sheets:
                if sheet_name != 'Structure Visualization':
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:Z', 15)  # Set column width to accommodate units
                    
                    # Format the header row
                    first_row = 0
                    if sheet_name == 'Units' or sheet_name == 'Summary':
                        # These sheets have an extra row for index name
                        first_row = 1
                        
                    # Get the last column with data - use a more conservative approach
                    last_col = 15  # Reasonable number for our data
                    try:
                        worksheet.conditional_format(first_row, 0, first_row, last_col, 
                                                   {'type': 'no_errors', 'format': header_format})
                    except:
                        pass  # Skip formatting if it fails
        
    print(f"Excel file created successfully: {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    json_file = "example_6_3/input_imperial_results_display.json"
    excel_file = json_to_excel(json_file)