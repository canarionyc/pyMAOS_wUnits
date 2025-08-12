# Interactive stairway structure visualization and editing tool

import os
import json
import sys
import re
import matplotlib
# Force use of TkAgg backend which has better interactive support
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import numpy as np
import datetime

# Import proj3d for 3D to 2D projections
try:
    from mpl_toolkits.mplot3d import proj3d
except ImportError:
    print("Warning: Could not import proj3d, some 3D functions may be limited")

# Global debug flag - set to True for verbose output
DEBUG = True

def debug_print(*args, **kwargs):
    """Helper function for debug printing that respects the global DEBUG flag"""
    if DEBUG:
        print(*args, **kwargs)

class StairwayInteractiveVisualizer:
    def __init__(self, json_file):
        debug_print(f"Initializing visualizer with file: {json_file}")
        self.json_file = json_file
        self.load_data()

        # Initialize state variables
        self.selected_node = None
        self.selected_member = None
        self.mode = 'view'  # 'view', 'move', 'add_node', 'add_member', 'delete_node', 'delete_member', 'edit_member'

        # For member creation
        self.member_creation_nodes = []
        self.next_member_id = max([m['id'] for m in self.members], default=0) + 1

        # Create the figure and 3D axes
        self.fig = plt.figure(figsize=(16, 10))
        debug_print("Figure created")
        self.ax = self.fig.add_subplot(111, projection='3d')
        debug_print("3D axes created")

        # Setup UI before connecting events
        self.setup_ui()
        debug_print("UI setup complete")

        # Connect event handlers
        debug_print("Connecting event handlers")
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw_cid = self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        debug_print(f"Event handlers connected: {self.click_cid}, {self.motion_cid}, {self.key_cid}, {self.draw_cid}")

        # Plot the stairway structure last
        self.plot_stairway()

        # Show debug message when initialization is complete
        debug_print("Visualizer initialization complete")

    def on_draw(self, event):
        """Debug handler for draw events"""
        debug_print("Draw event occurred")

    def load_data(self):
        """Load stairway data from JSON file"""
        print(f"Loading stairway structure from {self.json_file}")

        try:
            # Remove comments if present in the JSON file
            with open(os.path.join(os.path.dirname(__file__), self.json_file), 'r') as file:
                json_str = ""
                for line in file:
                    if '//' not in line:
                        json_str += line

                self.stairway_data = json.loads(json_str)

            # Extract structure data
            self.nodes = self.stairway_data['nodes']
            self.supports = self.stairway_data['supports']
            self.members = self.stairway_data['members']
            self.materials = self.stairway_data.get('materials', [])
            self.sections = self.stairway_data.get('sections', [])

            # Create node dictionary for quick lookups
            self.node_dict = {}
            for node in self.nodes:
                node_id = node['id']
                x = float(node['x'].split()[0])
                y = float(node['y'].split()[0])
                z = float(node['z'].split()[0])
                self.node_dict[node_id] = (x, y, z)

            # Create dictionaries for materials and sections
            self.material_dict = {m['id']: m for m in self.materials}
            self.section_dict = {s['id']: s for s in self.sections}

            # Create set of supported nodes
            self.supported_nodes = {s['node'] for s in self.supports}

            # Create map of connected members for each node
            self.node_connections = {}
            for node_id in self.node_dict:
                self.node_connections[node_id] = []

            for member in self.members:
                i_node = member['i_node']
                j_node = member['j_node']
                member_id = member['id']

                if i_node in self.node_connections:
                    self.node_connections[i_node].append(member_id)
                if j_node in self.node_connections:
                    self.node_connections[j_node].append(member_id)

            print(f"Successfully loaded: {self.stairway_data['design_parameters']['name']}")
            print(f"Structure contains {len(self.nodes)} nodes, {len(self.members)} members, and {len(self.supports)} supports")

        except Exception as e:
            print(f"Error loading JSON file: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def setup_ui(self):
        """Set up the UI elements for interactive editing"""
        # Create a smaller subplot for buttons
        self.button_area = plt.axes([0.01, 0.01, 0.20, 0.20])
        self.button_area.axis('off')

        # Mode selection radio buttons - add member modes
        self.mode_radio_ax = plt.axes([0.01, 0.80, 0.15, 0.15])
        self.mode_radio = RadioButtons(
            self.mode_radio_ax,
            ['View', 'Move Node', 'Add Node', 'Add Member', 'Delete Node', 'Delete Member', 'Edit Member'],
            activecolor='green'
        )
        self.mode_radio.on_clicked(self.set_mode)

        # Section selection for new members
        self.section_radio_ax = plt.axes([0.01, 0.55, 0.15, 0.10])
        section_labels = [f"Section {i}" for i in range(1, 6)]
        self.section_radio = RadioButtons(
            self.section_radio_ax,
            section_labels,
            activecolor='blue'
        )
        self.section_radio.on_clicked(lambda label: debug_print(f"Section changed to: {label}"))

        # Save button
        self.save_button_ax = plt.axes([0.01, 0.70, 0.15, 0.05])
        self.save_button = Button(self.save_button_ax, 'Save Structure')
        self.save_button.on_clicked(self.save_structure)

        # Reset view button
        self.reset_view_ax = plt.axes([0.01, 0.65, 0.15, 0.05])
        self.reset_view_button = Button(self.reset_view_ax, 'Reset View')
        self.reset_view_button.on_clicked(self.reset_view)

        # Information panel
        self.info_ax = plt.axes([0.01, 0.05, 0.20, 0.45])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(
            0, 1,
            "Interactive Stairway Editor\n\nClick on a node to select it\n"
            "Use the mode buttons to change actions\n"
            "Hotkeys: v=View, m=Move, a=Add, d=Delete, e=Edit",
            va='top'
        )

        # Node editor panel (initially hidden)
        self.node_editor_ax = plt.axes([0.85, 0.65, 0.14, 0.30])
        self.node_editor_ax.axis('off')
        self.node_editor_title = self.node_editor_ax.text(0, 1.1, "Node Editor", fontsize=10, weight='bold')
        self.node_editor_ax.set_visible(False)

        # X coordinate slider
        self.x_slider_ax = plt.axes([0.87, 0.85, 0.10, 0.03])
        self.x_slider = Slider(self.x_slider_ax, 'X', 0, 30, valinit=0)
        self.x_slider.on_changed(self.update_node_position)
        self.x_slider_ax.set_visible(False)

        # Y coordinate slider
        self.y_slider_ax = plt.axes([0.87, 0.80, 0.10, 0.03])
        self.y_slider = Slider(self.y_slider_ax, 'Y', 0, 10, valinit=0)
        self.y_slider.on_changed(self.update_node_position)
        self.y_slider_ax.set_visible(False)

        # Z coordinate slider
        self.z_slider_ax = plt.axes([0.87, 0.75, 0.10, 0.03])
        self.z_slider = Slider(self.z_slider_ax, 'Z', 0, 10, valinit=0)
        self.z_slider.on_changed(self.update_node_position)
        self.z_slider_ax.set_visible(False)

        # Member editor panel (initially hidden)
        self.member_editor_ax = plt.axes([0.85, 0.35, 0.14, 0.25])
        self.member_editor_ax.axis('off')
        self.member_editor_title = self.member_editor_ax.text(0, 1.1, "Member Editor", fontsize=10, weight='bold')
        self.member_editor_ax.set_visible(False)

        # Member section radio buttons
        self.member_section_ax = plt.axes([0.87, 0.45, 0.10, 0.10])
        self.member_section_radio = RadioButtons(
            self.member_section_ax,
            [f"Section {i}" for i in range(1, 6)],
            activecolor='blue'
        )
        self.member_section_radio.on_clicked(self.update_member_section)
        self.member_section_ax.set_visible(False)

        # Member material radio buttons
        self.member_material_ax = plt.axes([0.87, 0.30, 0.10, 0.10])
        self.member_material_radio = RadioButtons(
            self.member_material_ax,
            [f"Material {i}" for i in range(1, 4)],
            activecolor='orange'
        )
        self.member_material_radio.on_clicked(self.update_member_material)
        self.member_material_ax.set_visible(False)

        # Change member nodes button
        self.change_nodes_ax = plt.axes([0.87, 0.25, 0.10, 0.04])
        self.change_nodes_button = Button(self.change_nodes_ax, 'Change Nodes')
        self.change_nodes_button.on_clicked(self.start_change_member_nodes)
        self.change_nodes_ax.set_visible(False)

    def set_mode(self, mode_label):
        """Change the current editing mode"""
        debug_print(f"\n----- Mode change: {mode_label} -----")
        mode_map = {
            'View': 'view',
            'Move Node': 'move',
            'Add Node': 'add_node',
            'Add Member': 'add_member',
            'Delete Node': 'delete_node',
            'Delete Member': 'delete_member',
            'Edit Member': 'edit_member'
        }
        self.mode = mode_map[mode_label]
        debug_print(f"Mode changed to: {self.mode}")

        # Reset state when changing modes
        if self.mode == 'add_member':
            self.member_creation_nodes = []
            debug_print("Member creation state reset - select two nodes to create a member")
        elif self.mode == 'change_member_nodes':
            self.member_creation_nodes = []
            debug_print("Member node change state reset - select two nodes to reconnect the member")

        # Hide/show appropriate editors
        self.hide_node_editor()
        self.hide_member_editor()

        if self.mode == 'move' and self.selected_node:
            self.show_node_editor()
        elif self.mode == 'edit_member' and self.selected_member:
            self.show_member_editor()

        # Handle mode-specific actions
        if self.mode == 'delete_node' and self.selected_node:
            debug_print(f"Deleting node {self.selected_node}")
            self.delete_node()
        elif self.mode == 'delete_member' and self.selected_member:
            debug_print(f"Deleting member {self.selected_member}")
            self.delete_member()

        self.update_info_text()
        self.plot_stairway()  # Redraw to show/hide appropriate elements
        debug_print("Calling plt.draw() after mode change")
        plt.draw()

    def update_info_text(self):
        """Update the information panel text based on current state"""
        info = f"Mode: {self.mode.capitalize().replace('_', ' ')}\n\n"

        if self.mode == 'add_member':
            info += f"Creating new member - "
            if len(self.member_creation_nodes) == 0:
                info += "Select first node\n"
            elif len(self.member_creation_nodes) == 1:
                info += f"Node {self.member_creation_nodes[0]} selected\nSelect second node\n"
        elif self.mode == 'change_member_nodes':
            info += f"Changing member {self.selected_member} nodes - "
            if len(self.member_creation_nodes) == 0:
                info += "Select first node\n"
            elif len(self.member_creation_nodes) == 1:
                info += f"Node {self.member_creation_nodes[0]} selected\nSelect second node\n"

        if self.selected_node:
            node_pos = self.node_dict[self.selected_node]
            info += f"Selected Node: {self.selected_node}\n"
            info += f"Position: ({node_pos[0]:.2f}, {node_pos[1]:.2f}, {node_pos[2]:.2f})\n"
            info += f"Connected to {len(self.node_connections[self.selected_node])} members\n"

            if self.selected_node in self.supported_nodes:
                info += "This node has support constraints\n"
        elif self.selected_member:
            member = next((m for m in self.members if m['id'] == self.selected_member), None)
            if member:
                info += f"Selected Member: {self.selected_member}\n"
                info += f"Connects nodes: {member['i_node']} - {member['j_node']}\n"
                info += f"Section: {member['section']}, Material: {member['material']}\n"

                # Get member length
                i_node = member['i_node']
                j_node = member['j_node']
                if i_node in self.node_dict and j_node in self.node_dict:
                    x1, y1, z1 = self.node_dict[i_node]
                    x2, y2, z2 = self.node_dict[j_node]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    info += f"Length: {length:.2f} ft\n"

        self.info_text.set_text(info)
        plt.draw()

    def plot_stairway(self):
        """Plot the stairway structure"""
        self.ax.clear()

        # Define colors and widths for different section types
        section_colors = {
            1: 'blue',    # Main stringers
            2: 'green',   # Steps/treads
            3: 'cyan',    # Secondary members
            4: 'gray',    # Bracing
            5: 'red'      # Column
        }

        section_widths = {
            1: 2,         # Main stringers
            2: 1.5,       # Steps/treads
            3: 1,         # Secondary members
            4: 1,         # Bracing
            5: 3          # Column
        }

        # Plot members
        self.member_lines = {}
        for member in self.members:
            i_node = member['i_node']
            j_node = member['j_node']
            member_id = member['id']
            section = member['section']

            if i_node not in self.node_dict or j_node not in self.node_dict:
                print(f"WARNING: Member {member_id} references non-existent node(s): {i_node}-{j_node}")
                continue

            x1, y1, z1 = self.node_dict[i_node]
            x2, y2, z2 = self.node_dict[j_node]

            color = section_colors.get(section, 'black')
            linewidth = section_widths.get(section, 1)

            # Highlight selected member
            if member_id == self.selected_member:
                color = 'red'
                linewidth += 1

            line = self.ax.plot([x1, x2], [y1, y2], [z1, z2],
                          color=color, linewidth=linewidth)[0]
            self.member_lines[member_id] = line

        # Plot nodes
        self.node_points = {}
        for node_id, (x, y, z) in self.node_dict.items():
            # Different marker for nodes in the member creation process
            if node_id in self.member_creation_nodes:
                point = self.ax.scatter(x, y, z, color='blue', s=60, marker='o', edgecolors='black')
            else:
                point = self.ax.scatter(x, y, z, color='black', s=20)
            self.node_points[node_id] = point

        # Plot supports
        for support in self.supports:
            node_id = support['node']
            if node_id not in self.node_dict:
                print(f"WARNING: Support references non-existent node: {node_id}")
                continue

            x, y, z = self.node_dict[node_id]

            # Full support (fixed in all directions)
            if support['ux'] == 1 and support['uy'] == 1 and support['uz'] == 1:
                self.ax.scatter(x, y, z, color='red', s=100, marker='s')
            # Partial support
            else:
                self.ax.scatter(x, y, z, color='orange', s=80, marker='^')

        # Add a wall surface to represent the back wall
        # Find the y value for the back of the stairway
        back_y = max(pos[1] for pos in self.node_dict.values())
        min_x = min(pos[0] for pos in self.node_dict.values())
        max_x = max(pos[0] for pos in self.node_dict.values())
        min_z = min(pos[2] for pos in self.node_dict.values())
        max_z = max(pos[2] for pos in self.node_dict.values())

        # Add some margin to the wall
        margin = 1
        wall_x = np.array([min_x-margin, max_x+margin])
        wall_z = np.array([min_z-margin, max_z+margin])
        wall_x_grid, wall_z_grid = np.meshgrid(wall_x, wall_z)
        wall_y_grid = np.ones_like(wall_x_grid) * back_y

        # Plot the wall with slight transparency
        self.ax.plot_surface(wall_x_grid, wall_y_grid, wall_z_grid, color='gray', alpha=0.2)

        # Create a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=section_colors[1], lw=section_widths[1], label='Main Stringers'),
            Line2D([0], [0], color=section_colors[2], lw=section_widths[2], label='Steps/Treads'),
            Line2D([0], [0], color=section_colors[4], lw=section_widths[4], label='Bracing'),
            Line2D([0], [0], color=section_colors[5], lw=section_widths[5], label='Column'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Fixed Support'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='Partial Support')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        # Set axis labels and title
        self.ax.set_xlabel('X (ft)')
        self.ax.set_ylabel('Y (ft)')
        self.ax.set_zlabel('Z (ft)')
        title = self.stairway_data['design_parameters']['name']
        self.ax.set_title(title)

        # Adjust view angle for better visualization
        self.ax.view_init(elev=20, azim=-60)

        # Set equal aspect ratio for a more realistic view
        self.ax.set_box_aspect([max_x-min_x, back_y+1, max_z-min_z])

        # Add text annotations for key structural elements
        self.ax.text(0, 0, max_z+0.5, "Upper Landing", color='blue')
        self.ax.text(max_x, 0, min_z-0.5, "Lower Landing", color='blue')

        # Highlight selected node if any
        if self.selected_node and self.selected_node in self.node_dict:
            x, y, z = self.node_dict[self.selected_node]
            self.ax.scatter(x, y, z, color='red', s=100, marker='o', edgecolors='black')

    def find_closest_node(self, event):
        """Find the closest node to the clicked position"""
        debug_print("\n===== Finding closest node =====")
        if not event.inaxes:
            debug_print("Click not in any axes")
            return None

        if event.inaxes != self.ax:
            debug_print(f"Click in wrong axes: {event.inaxes}")
            return None

        # Get screen coordinates of all nodes
        closest_node = None
        min_dist = float('inf')

        # Raw click coordinates
        click_x, click_y = event.x, event.y
        debug_print(f"Raw click at screen coordinates: ({click_x}, {click_y})")

        try:
            # Directly use screen coordinates with imported proj3d module
            from mpl_toolkits.mplot3d import proj3d

            for node_id, (x, y, z) in self.node_dict.items():
                try:
                    # Convert 3D coordinates to display coordinates
                    xs, ys, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())

                    # Convert to display coordinates
                    display_coords = self.ax.transData.transform([(xs, ys)])
                    if len(display_coords) > 0:
                        sx, sy = display_coords[0]
                        dist = np.sqrt((click_x - sx)**2 + (click_y - sy)**2)

                        debug_print(f"Node {node_id} at 3D=({x:.1f},{y:.1f},{z:.1f}), "
                                   f"screen=({sx:.1f},{sy:.1f}), dist={dist:.1f}")

                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node_id
                except Exception as e:
                    debug_print(f"Error projecting node {node_id}: {e}")
                    continue

        except ImportError:
            # Fallback method if proj3d import fails
            debug_print("Using fallback node detection method")

            # Get view angles
            elev, azim = self.ax.elev, self.ax.azim

            # Convert azimuth from degrees to radians
            azim_rad = np.radians(azim)
            elev_rad = np.radians(elev)

            for node_id, (x, y, z) in self.node_dict.items():
                try:
                    # Simple projection based on view angles
                    # This is an approximation but works reasonably for most views
                    sx = x * np.cos(azim_rad) - y * np.sin(azim_rad)
                    sy = z * np.cos(elev_rad) + np.sin(elev_rad) * (x * np.sin(azim_rad) + y * np.cos(azim_rad))

                    # Convert to display coordinates (simplified)
                    display_coords = self.ax.transData.transform([(sx, sy)])
                    if len(display_coords) > 0:
                        disp_x, disp_y = display_coords[0]
                        dist = np.sqrt((click_x - disp_x)**2 + (click_y - disp_y)**2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node_id
                except Exception as e:
                    debug_print(f"Error in fallback projection for node {node_id}: {e}")
                    continue

        # Use a reasonable threshold (in pixels)
        threshold = 30
        if min_dist < threshold:
            debug_print(f"Found closest node: {closest_node} (distance: {min_dist:.1f}px)")
            return closest_node
        else:
            debug_print(f"No node within threshold. Closest was {min_dist:.1f}px away.")
            return None

    def find_closest_member(self, event):
        """Find the closest member to the clicked position"""
        debug_print("\n===== Finding closest member =====")
        if not event.inaxes or event.inaxes != self.ax:
            return None

        # Raw click coordinates
        click_x, click_y = event.x, event.y
        debug_print(f"Raw click at screen coordinates: ({click_x}, {click_y})")

        # Find closest member by measuring distance to line segments in screen coordinates
        closest_member = None
        min_dist = float('inf')

        for member in self.members:
            member_id = member['id']
            i_node = member['i_node']
            j_node = member['j_node']

            if i_node not in self.node_dict or j_node not in self.node_dict:
                continue

            # Get 3D coordinates
            x1, y1, z1 = self.node_dict[i_node]
            x2, y2, z2 = self.node_dict[j_node]

            try:
                # Project to 2D screen coordinates
                from mpl_toolkits.mplot3d import proj3d

                # Project endpoints
                xs1, ys1, _ = proj3d.proj_transform(x1, y1, z1, self.ax.get_proj())
                xs2, ys2, _ = proj3d.proj_transform(x2, y2, z2, self.ax.get_proj())

                # Transform to display coordinates
                display_coords1 = self.ax.transData.transform([(xs1, ys1)])
                display_coords2 = self.ax.transData.transform([(xs2, ys2)])

                if len(display_coords1) > 0 and len(display_coords2) > 0:
                    sx1, sy1 = display_coords1[0]
                    sx2, sy2 = display_coords2[0]

                    # Calculate distance from click to line segment
                    dist = point_to_line_distance(click_x, click_y, sx1, sy1, sx2, sy2)

                    debug_print(f"Member {member_id} distance: {dist:.1f}px")

                    if dist < min_dist:
                        min_dist = dist
                        closest_member = member_id
            except Exception as e:
                debug_print(f"Error projecting member {member_id}: {e}")
                continue

        # Use a reasonable threshold (in pixels)
        threshold = 15  # Smaller than node threshold
        if min_dist < threshold:
            debug_print(f"Found closest member: {closest_member} (distance: {min_dist:.1f}px)")
            return closest_member
        else:
            debug_print(f"No member within threshold. Closest was {min_dist:.1f}px away.")
            return None

    def on_click(self, event):
        """Handle mouse click events"""
        debug_print(f"\n***** Click event at ({event.x}, {event.y}) *****")

        # Check if click is in the main axes
        if event.inaxes != self.ax:
            debug_print(f"Click not in main axes but in {event.inaxes}")
            # Let the widgets handle their own events
            return

        # Handle different modes
        if self.mode == 'add_member':
            # In add_member mode, we select two nodes to create a member between them
            node_id = self.find_closest_node(event)
            if node_id:
                # If we already have this node in our selection, ignore the click
                if node_id in self.member_creation_nodes:
                    debug_print(f"Node {node_id} already selected for member creation")
                    return

                # Add the node to our member creation list
                self.member_creation_nodes.append(node_id)
                debug_print(f"Added node {node_id} to member creation (nodes: {self.member_creation_nodes})")

                # If we have two nodes, create the member
                if len(self.member_creation_nodes) == 2:
                    self.create_member(self.member_creation_nodes[0], self.member_creation_nodes[1])
                    # Reset for the next member
                    self.member_creation_nodes = []

                self.update_info_text()
                self.plot_stairway()
                plt.draw()

        elif self.mode == 'change_member_nodes':
            # In change_member_nodes mode, we select two nodes to reconnect an existing member
            node_id = self.find_closest_node(event)
            if node_id:
                # If we already have this node in our selection, ignore the click
                if node_id in self.member_creation_nodes:
                    debug_print(f"Node {node_id} already selected for member reconnection")
                    return

                # Add the node to our member creation list
                self.member_creation_nodes.append(node_id)
                debug_print(f"Added node {node_id} to member reconnection (nodes: {self.member_creation_nodes})")

                # If we have two nodes, update the member
                if len(self.member_creation_nodes) == 2:
                    self.update_member_nodes(self.selected_member,
                                           self.member_creation_nodes[0],
                                           self.member_creation_nodes[1])
                    # Reset state and mode
                    self.member_creation_nodes = []
                    self.mode = 'edit_member'
                    self.mode_radio.set_active(6)  # Select 'Edit Member' in the UI

                self.update_info_text()
                self.plot_stairway()
                plt.draw()

        elif self.mode == 'delete_member':
            # Try to find a member to delete
            member_id = self.find_closest_member(event)
            if member_id:
                debug_print(f"Selected member {member_id} for deletion")
                self.selected_member = member_id
                self.delete_member()
                self.update_info_text()
                self.plot_stairway()
                plt.draw()

        elif self.mode.startswith('delete_node'):
            # Find closest node to the click
            node_id = self.find_closest_node(event)
            if node_id:
                debug_print(f"Selected node: {node_id} for deletion")
                self.selected_node = node_id
                self.delete_node()
                self.update_info_text()
                self.plot_stairway()
                plt.draw()

        elif self.mode == 'edit_member':
            # Try to find a member to edit
            member_id = self.find_closest_member(event)
            if member_id:
                debug_print(f"Selected member {member_id} for editing")
                self.selected_member = member_id
                self.selected_node = None
                self.show_member_editor()
                self.update_info_text()
                self.plot_stairway()
                plt.draw()

        else:  # View or other modes
            # Try to find a node first
            node_id = self.find_closest_node(event)
            if node_id:
                debug_print(f"Selected node: {node_id}")
                self.selected_node = node_id
                self.selected_member = None

                # If in move mode, show the position sliders
                if self.mode == 'move':
                    debug_print(f"Move mode active - showing node editor for node {node_id}")
                    self.show_node_editor()

                self.update_info_text()
                self.plot_stairway()
                plt.draw()
            else:
                # If no node found, try to find a member
                member_id = self.find_closest_member(event)
                if member_id:
                    debug_print(f"Selected member: {member_id}")
                    self.selected_member = member_id
                    self.selected_node = None
                    self.update_info_text()
                    self.plot_stairway()
                    plt.draw()
                else:
                    # Clicked empty space - deselect everything
                    if self.selected_node or self.selected_member:
                        debug_print("Deselecting current selection")
                        self.selected_node = None
                        self.selected_member = None
                        self.hide_node_editor()
                        self.hide_member_editor()
                        self.update_info_text()
                        self.plot_stairway()
                        plt.draw()

    def on_motion(self, event):
        """Handle mouse motion events"""
        # Simple implementation to avoid AttributeError
        if not event.inaxes or event.inaxes != self.ax:
            return

        # Could implement hover highlighting here
        pass

    def on_key(self, event):
        """Handle keyboard events"""
        debug_print(f"\n##### Key press: {event.key} #####")

        if event.key == 'v':
            debug_print("Switching to View mode")
            self.mode_radio.set_active(0)
            self.mode = 'view'
            self.hide_node_editor()
            self.hide_member_editor()
        elif event.key == 'm':
            debug_print("Switching to Move mode")
            self.mode_radio.set_active(1)
            self.mode = 'move'
            if self.selected_node:
                self.show_node_editor()
                self.hide_member_editor()
        elif event.key == 'a':
            debug_print("Switching to Add Node mode")
            self.mode_radio.set_active(2)
            self.mode = 'add_node'
            self.hide_node_editor()
            self.hide_member_editor()
        elif event.key == 'shift+a':
            debug_print("Switching to Add Member mode")
            self.mode_radio.set_active(3)
            self.mode = 'add_member'
            self.member_creation_nodes = []
            self.hide_node_editor()
            self.hide_member_editor()
        elif event.key == 'd':
            debug_print("Switching to Delete Node mode")
            self.mode_radio.set_active(4)
            self.mode = 'delete_node'
            self.hide_node_editor()
            self.hide_member_editor()
            if self.selected_node and self.mode == 'delete_node':
                self.delete_node()
        elif event.key == 'shift+d':
            debug_print("Switching to Delete Member mode")
            self.mode_radio.set_active(5)
            self.mode = 'delete_member'
            self.hide_node_editor()
            self.hide_member_editor()
            if self.selected_member and self.mode == 'delete_member':
                self.delete_member()
        elif event.key == 'e':
            debug_print("Switching to Edit Member mode")
            self.mode_radio.set_active(6)
            self.mode = 'edit_member'
            self.hide_node_editor()
            if self.selected_member:
                self.show_member_editor()
        elif event.key == 'escape':
            debug_print("Escape pressed - deselecting")
            self.selected_node = None
            self.selected_member = None
            self.member_creation_nodes = []
            self.hide_node_editor()
            self.hide_member_editor()
            self.plot_stairway()  # Redraw without selection highlight

        self.update_info_text()
        plt.draw()

    def show_node_editor(self):
        """Show the node editor panel with current node position"""
        if not self.selected_node:
            debug_print("Cannot show node editor: no node selected")
            return

        debug_print(f"Showing node editor for node {self.selected_node}")
        x, y, z = self.node_dict[self.selected_node]

        # Adjust slider ranges to encompass the current value
        self.x_slider.valmin = max(0, x - 10)
        self.x_slider.valmax = x + 10
        self.y_slider.valmin = max(0, y - 5)
        self.y_slider.valmax = y + 5
        self.z_slider.valmin = max(0, z - 5)
        self.z_slider.valmax = z + 5

        # Update slider values
        self.x_slider.set_val(x)
        self.y_slider.set_val(y)
        self.z_slider.set_val(z)

        # Make the editor visible
        debug_print("Setting editor UI elements to visible")
        self.node_editor_ax.set_visible(True)
        self.x_slider_ax.set_visible(True)
        self.y_slider_ax.set_visible(True)
        self.z_slider_ax.set_visible(True)
        debug_print("Calling plt.draw() to update visibility")
        plt.draw()

    def show_member_editor(self):
        """Show the member editor panel with current member properties"""
        if not self.selected_member:
            debug_print("Cannot show member editor: no member selected")
            return

        debug_print(f"Showing member editor for member {self.selected_member}")

        # Get current member properties
        member = next((m for m in self.members if m['id'] == self.selected_member), None)
        if not member:
            debug_print(f"Member {self.selected_member} not found in members list")
            return

        # Set the radio button selections to match current properties
        section_idx = member['section'] - 1  # Adjust for 0-based indexing
        material_idx = member['material'] - 1  # Adjust for 0-based indexing

        # Set active radio buttons
        for i, circle in enumerate(self.member_section_radio.circles):
            circle.set_facecolor('white')
        self.member_section_radio.circles[section_idx].set_facecolor('blue')

        for i, circle in enumerate(self.member_material_radio.circles):
            circle.set_facecolor('white')
        self.member_material_radio.circles[material_idx].set_facecolor('orange')

        # Make the editor visible
        self.member_editor_ax.set_visible(True)
        self.member_section_ax.set_visible(True)
        self.member_material_ax.set_visible(True)
        self.change_nodes_ax.set_visible(True)

        plt.draw()

    def hide_node_editor(self):
        """Hide the node editor panel"""
        debug_print("Hiding node editor")
        self.node_editor_ax.set_visible(False)
        self.x_slider_ax.set_visible(False)
        self.y_slider_ax.set_visible(False)
        self.z_slider_ax.set_visible(False)
        plt.draw()

    def hide_member_editor(self):
        """Hide the member editor panel"""
        debug_print("Hiding member editor")
        self.member_editor_ax.set_visible(False)
        self.member_section_ax.set_visible(False)
        self.member_material_ax.set_visible(False)
        self.change_nodes_ax.set_visible(False)
        plt.draw()

    def update_node_position(self, val):
        """Update the selected node's position from slider values"""
        if not self.selected_node:
            debug_print("Cannot update position: no node selected")
            return

        # Get new position from sliders
        x = self.x_slider.val
        y = self.y_slider.val
        z = self.z_slider.val

        debug_print(f"Updating node {self.selected_node} position to ({x:.2f}, {y:.2f}, {z:.2f})")

        # Update node position
        self.node_dict[self.selected_node] = (x, y, z)

        # Update position in the nodes list too
        for node in self.nodes:
            if node['id'] == self.selected_node:
                node['x'] = f"{x} ft"
                node['y'] = f"{y} ft"
                node['z'] = f"{z} ft"
                break

        # Redraw structure
        debug_print("Redrawing structure after position update")
        self.plot_stairway()
        self.update_info_text()

    def update_member_section(self, label):
        """Update the selected member's section type"""
        if not self.selected_member:
            debug_print("Cannot update section: no member selected")
            return

        section = int(label.split()[-1])  # Extract number from "Section X"
        debug_print(f"Updating member {self.selected_member} section to {section}")

        # Update member section
        for member in self.members:
            if member['id'] == self.selected_member:
                member['section'] = section
                break

        # Redraw structure
        self.plot_stairway()
        self.update_info_text()

    def update_member_material(self, label):
        """Update the selected member's material"""
        if not self.selected_member:
            debug_print("Cannot update material: no member selected")
            return

        material = int(label.split()[-1])  # Extract number from "Material X"
        debug_print(f"Updating member {self.selected_member} material to {material}")

        # Update member material
        for member in self.members:
            if member['id'] == self.selected_member:
                member['material'] = material
                break

        # Redraw structure
        self.plot_stairway()
        self.update_info_text()

    def start_change_member_nodes(self, event):
        """Start the process of changing a member's nodes"""
        if not self.selected_member:
            debug_print("Cannot change nodes: no member selected")
            return

        debug_print(f"Starting node change for member {self.selected_member}")
        self.mode = 'change_member_nodes'
        self.member_creation_nodes = []
        self.update_info_text()

    def update_member_nodes(self, member_id, new_i_node, new_j_node):
        """Update a member's connected nodes"""
        member = next((m for m in self.members if m['id'] == member_id), None)
        if not member:
            debug_print(f"Member {member_id} not found")
            return

        old_i_node = member['i_node']
        old_j_node = member['j_node']

        debug_print(f"Updating member {member_id} nodes from {old_i_node}-{old_j_node} to {new_i_node}-{new_j_node}")

        # Remove member from old node connections
        if old_i_node in self.node_connections and member_id in self.node_connections[old_i_node]:
            self.node_connections[old_i_node].remove(member_id)

        if old_j_node in self.node_connections and member_id in self.node_connections[old_j_node]:
            self.node_connections[old_j_node].remove(member_id)

        # Update member endpoints
        member['i_node'] = new_i_node
        member['j_node'] = new_j_node

        # Add to new node connections
        if new_i_node not in self.node_connections:
            self.node_connections[new_i_node] = []
        self.node_connections[new_i_node].append(member_id)

        if new_j_node not in self.node_connections:
            self.node_connections[new_j_node] = []
        self.node_connections[new_j_node].append(member_id)

        debug_print(f"Member {member_id} nodes updated successfully")

    def delete_node(self):
        """Delete the selected node and connected members"""
        if not self.selected_node:
            print("Cannot delete: no node selected")
            return

        print(f"Attempting to delete node {self.selected_node}")

        # Check if node has any supports
        if self.selected_node in self.supported_nodes:
            print(f"Cannot delete node {self.selected_node} because it has support constraints.")
            return

        # Get connected members
        connected_members = self.node_connections[self.selected_node].copy()  # Make a copy to avoid modification during iteration
        if connected_members:
            print(f"Deleting {len(connected_members)} members connected to node {self.selected_node}")

            # Delete members
            self.members = [m for m in self.members
                           if m['id'] not in connected_members]

            # Update connections dict for affected nodes
            for member_id in connected_members:
                member = next((m for m in self.members if m['id'] == member_id), None)
                if member:
                    other_node = member['i_node'] if member['j_node'] == self.selected_node else member['j_node']
                    if other_node in self.node_connections and member_id in self.node_connections[other_node]:
                        self.node_connections[other_node].remove(member_id)

        # Delete node
        self.nodes = [n for n in self.nodes if n['id'] != self.selected_node]
        if self.selected_node in self.node_dict:
            del self.node_dict[self.selected_node]
        if self.selected_node in self.node_connections:
            del self.node_connections[self.selected_node]

        # Clear selection
        self.selected_node = None
        self.hide_node_editor()

        # Redraw structure
        self.plot_stairway()
        self.update_info_text()

    def delete_member(self):
        """Delete the selected member"""
        if not self.selected_member:
            debug_print("Cannot delete: no member selected")
            return

        member_id = self.selected_member
        debug_print(f"Deleting member {member_id}")

        # Find the member
        member = next((m for m in self.members if m['id'] == member_id), None)
        if not member:
            debug_print(f"Member {member_id} not found")
            return

        # Remove member from connections
        i_node = member['i_node']
        j_node = member['j_node']

        if i_node in self.node_connections and member_id in self.node_connections[i_node]:
            self.node_connections[i_node].remove(member_id)

        if j_node in self.node_connections and member_id in self.node_connections[j_node]:
            self.node_connections[j_node].remove(member_id)

        # Remove member from list
        self.members = [m for m in self.members if m['id'] != member_id]

        # Clear selection
        self.selected_member = None
        self.hide_member_editor()

        debug_print(f"Member {member_id} deleted")

    def reset_view(self, event):
        """Reset the 3D view to default angle"""
        print("Resetting view")
        self.ax.view_init(elev=20, azim=-60)
        plt.draw()

    def save_structure(self, event):
        """Save the modified structure back to a JSON file"""
        print("Saving structure...")
        # Update the structure data with modified nodes
        self.stairway_data['nodes'] = self.nodes
        self.stairway_data['members'] = self.members

        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(self.json_file)[0]
        output_file = f"{base_name}_edited_{timestamp}.json"

        try:
            with open(os.path.join(os.path.dirname(__file__), output_file), 'w') as f:
                json.dump(self.stairway_data, f, indent=4)
            print(f"Successfully saved structure to {output_file}")
        except Exception as e:
            print(f"Error saving structure: {e}")

    def create_member(self, node1, node2):
        """Create a new member between two nodes"""
        # Determine the section type from the radio button
        section_label = self.section_radio.value_selected
        section_number = int(section_label.split()[-1])

        # Create a new member
        new_member = {
            'id': self.next_member_id,
            'i_node': node1,
            'j_node': node2,
            'material': 1,  # Default material
            'section': section_number
        }

        debug_print(f"Creating new member: {new_member}")

        # Add to members list
        self.members.append(new_member)

        # Update connections dictionary
        self.node_connections[node1].append(self.next_member_id)
        self.node_connections[node2].append(self.next_member_id)

        # Increment ID counter for next member
        self.next_member_id += 1

        # Highlight the new member temporarily
        self.selected_member = new_member['id']

        # Provide feedback
        debug_print(f"Member {new_member['id']} created between nodes {node1} and {node2}")


# Helper functions for distance calculations
def point_to_line_distance(x, y, x1, y1, x2, y2):
    """Calculate the distance from point (x,y) to line segment (x1,y1)-(x2,y2)"""
    # Line segment length squared
    l2 = (x2 - x1)**2 + (y2 - y1)**2

    if l2 == 0:  # Line segment is a point
        return np.sqrt((x - x1)**2 + (y - y1)**2)

    # Calculate projection ratio (0-1 if within segment)
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))

    # Calculate closest point on segment
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    # Return distance to that point
    return np.sqrt((x - px)**2 + (y - py)**2)


# Main execution
if __name__ == "__main__":
    # Check if a file was provided as a command line argument
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default to a sample file if none specified
        json_file = "stairway_structure_Imperial.JSON"
        print(f"No file specified, using default: {json_file}")

        # Check if the file exists
        if not os.path.exists(os.path.join(os.path.dirname(__file__), json_file)):
            print(f"Default file {json_file} not found.")
            available_files = [f for f in os.listdir(os.path.dirname(__file__))
                              if f.endswith('.json')]

            if available_files:
                json_file = available_files[0]
                print(f"Using first available JSON file: {json_file}")
            else:
                print("No JSON files found in the current directory.")
                sys.exit(1)

    # Create and run the visualizer
    visualizer = StairwayInteractiveVisualizer(json_file)
    plt.show()
