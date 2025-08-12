# Interactive stairway structure visualization and editing tool using pure PyQt (no OpenGL)

import os
import json
import sys
import math  # Add math module import for sqrt function
import datetime
import traceback
import scipy as sp
from scipy.spatial import distance
# Import specific math functions from scipy
from scipy import special
from scipy import linalg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QSlider,
                            QGroupBox, QRadioButton, QButtonGroup, QSplitter,
                            QTextEdit, QFileDialog, QMessageBox, QSpinBox,
                            QDoubleSpinBox, QComboBox, QTreeWidget, QTreeWidgetItem,
                            QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QFont, QTransform

# Global debug flag
DEBUG = True

def debug_print(*args, **kwargs):
    """Helper function for debug printing"""
    if DEBUG:
        print(*args, **kwargs)
        sys.stdout.flush()  # Ensure output is flushed immediately

class StructureView(QGraphicsView):
    """Custom graphics view for 2D visualization of 3D structure"""

    nodeSelected = pyqtSignal(int)
    memberSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        debug_print("Initializing StructureView")
        super().__init__(parent)

        # Setup the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Enable antialiasing
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)

        # Set background color
        self.setBackgroundBrush(QBrush(QColor(245, 245, 245)))

        # View settings
        self.scale_factor = 20.0  # Scaling for better visibility
        self.center_offset = (0, 0)  # Pan offset

        # Structure data
        self.nodes = {}
        self.members = []
        self.supports = []

        # Selection
        self.selected_node = None
        self.selected_member = None

        # Colors for sections
        self.section_colors = {
            1: QColor(0, 0, 255),     # Blue - Main stringers
            2: QColor(0, 180, 0),     # Green - Steps/treads
            3: QColor(0, 200, 200),   # Cyan - Secondary members
            4: QColor(100, 100, 100), # Gray - Bracing
            5: QColor(200, 0, 0)      # Red - Column
        }

        # Mouse tracking for pan and zoom
        self.setMouseTracking(True)
        self.last_mouse_pos = None

        # Enable drag mode
        self.setDragMode(QGraphicsView.NoDrag)

        # Add axes display property
        self.show_axes = True

        # Use standard isometric projection angles for right-handed system
        # These are based on standard 30° isometric angles
        self.iso_angle = math.radians(30)  # Standard isometric angle
        self.cos_angle = math.cos(self.iso_angle)
        self.sin_angle = math.sin(self.iso_angle)

        debug_print("StructureView initialization complete")

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        try:
            debug_print("Wheel event")
            # Save the scene pos
            old_pos = self.mapToScene(event.pos())

            # Zoom factor
            zoom_factor = 1.15

            # Zoom in or out
            if event.angleDelta().y() > 0:
                self.scale(zoom_factor, zoom_factor)
            else:
                self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

            # Get the new position
            new_pos = self.mapToScene(event.pos())

            # Move scene to old position
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        except Exception as e:
            debug_print(f"Error in wheelEvent: {str(e)}")
            traceback.print_exc()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        try:
            debug_print(f"Mouse press at {event.pos().x()}, {event.pos().y()}")
            self.last_mouse_pos = event.pos()

            if event.button() == Qt.LeftButton:
                # Convert to scene coordinates
                scene_pos = self.mapToScene(event.pos())
                debug_print(f"Scene position: {scene_pos.x()}, {scene_pos.y()}")

                # Try to find closest node
                try:
                    node_id = self.find_closest_node(scene_pos.x(), scene_pos.y())
                    if node_id is not None:
                        debug_print(f"Selected node {node_id}")
                        self.selected_node = node_id
                        self.selected_member = None
                        self.nodeSelected.emit(node_id)
                        self.update_scene()
                        return
                except Exception as e:
                    debug_print(f"Error finding closest node: {str(e)}")
                    traceback.print_exc()

                # Try to find closest member
                try:
                    member_id = self.find_closest_member(scene_pos.x(), scene_pos.y())
                    if member_id is not None:
                        debug_print(f"Selected member {member_id}")
                        self.selected_member = member_id
                        self.selected_node = None
                        self.memberSelected.emit(member_id)
                        self.update_scene()
                        return
                except Exception as e:
                    debug_print(f"Error finding closest member: {str(e)}")
                    traceback.print_exc()

                # If clicked on empty space
                self.selected_node = None
                self.selected_member = None
                self.update_scene()
        except Exception as e:
            debug_print(f"Error in mousePressEvent: {str(e)}")
            traceback.print_exc()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for panning"""
        try:
            if event.buttons() & Qt.MiddleButton and self.last_mouse_pos:
                # Calculate movement
                delta = event.pos() - self.last_mouse_pos
                self.last_mouse_pos = event.pos()

                # Pan the view
                self.translate(delta.x(), delta.y())
        except Exception as e:
            debug_print(f"Error in mouseMoveEvent: {str(e)}")
            traceback.print_exc()

        super().mouseMoveEvent(event)

    def find_closest_node(self, x, y, threshold=10):
        """Find the closest node to the given coordinates"""
        debug_print(f"Finding closest node to {x}, {y}")

        try:
            closest_node = None
            min_dist = threshold

            # Safety check for nodes
            if not self.nodes:
                debug_print("No nodes available")
                return None

            debug_print(f"Checking {len(self.nodes)} nodes")
            for node_id, coords in self.nodes.items():
                if not coords or len(coords) != 3:
                    debug_print(f"Invalid node data for node {node_id}: {coords}")
                    continue

                nx, ny, nz = coords
                if not isinstance(nx, (int, float)) or not isinstance(ny, (int, float)) or not isinstance(nz, (int, float)):
                    debug_print(f"Non-numeric coordinates for node {node_id}: {coords}")
                    continue

                # Project 3D point to 2D using simple projection
                try:
                    proj_x, proj_y = self.project_3d_to_2d(nx, ny, nz)

                    # Calculate distance - using math.sqrt instead of sp.sqrt
                    try:
                        dist = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node_id
                    except Exception as e:
                        debug_print(f"Error calculating distance for node {node_id}: {str(e)}")
                except Exception as e:
                    debug_print(f"Error projecting node {node_id}: {str(e)}")

            if closest_node is not None:
                debug_print(f"Found closest node: {closest_node} at distance {min_dist:.2f}")
            else:
                debug_print("No node found within threshold")

            return closest_node

        except Exception as e:
            debug_print(f"Error in find_closest_node: {str(e)}")
            traceback.print_exc()
            return None

    def find_closest_member(self, x, y, threshold=5):
        """Find the closest member to the given coordinates"""
        debug_print(f"Finding closest member to {x}, {y}")

        try:
            closest_member = None
            min_dist = threshold

            # Safety check for members
            if not self.members:
                debug_print("No members available")
                return None

            for member in self.members:
                try:
                    i_node = member.get('i_node')
                    j_node = member.get('j_node')

                    if i_node is None or j_node is None:
                        debug_print(f"Missing node references in member {member.get('id')}")
                        continue

                    if i_node not in self.nodes or j_node not in self.nodes:
                        debug_print(f"Member {member.get('id')} refers to non-existent node(s)")
                        continue

                    # Project endpoints
                    x1, y1, z1 = self.nodes[i_node]
                    x2, y2, z2 = self.nodes[j_node]

                    p1x, p1y = self.project_3d_to_2d(x1, y1, z1)
                    p2x, p2y = self.project_3d_to_2d(x2, y2, z2)

                    # Calculate distance to line segment
                    dist = self.point_to_line_dist(x, y, p1x, p1y, p2x, p2y)

                    if dist < min_dist:
                        min_dist = dist
                        closest_member = member['id']
                except Exception as e:
                    debug_print(f"Error processing member {member.get('id')}: {str(e)}")

            return closest_member
        except Exception as e:
            debug_print(f"Error in find_closest_member: {str(e)}")
            traceback.print_exc()
            return None

    def point_to_line_dist(self, x, y, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        try:
            # Calculate line segment length squared
            l2 = (x2 - x1)**2 + (y2 - y1)**2

            if l2 == 0:  # Points are the same
                return math.sqrt((x - x1)**2 + (y - y1)**2)

            # Calculate projection parameter
            t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2
            t = max(0, min(1, t))  # Clamp to segment

            # Calculate nearest point on segment
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)

            # Return distance to that point
            return math.sqrt((x - px)**2 + (y - py)**2)
        except Exception as e:
            debug_print(f"Error in point_to_line_dist: {str(e)}")
            return float('inf')  # Return infinite distance on error

    def project_3d_to_2d(self, x, y, z):
        """Project 3D point to 2D using a standard isometric projection with right-handed coordinate system"""
        try:
            # Implement standard isometric projection for right-handed system
            # X right, Y forward (into screen), Z up
            # When rotating from X to Y with right-hand rule, thumb points up along Z

            # Standard isometric projection coefficients (30° angles)
            proj_x = x * self.cos_angle - y * self.cos_angle  # X and Y contributions to screen X
            proj_y = z + x * self.sin_angle + y * self.sin_angle  # Z, X and Y contributions to screen Y

            # Apply scale and offset
            proj_x = proj_x * self.scale_factor + 500 + self.center_offset[0]
            proj_y = -proj_y * self.scale_factor + 300 + self.center_offset[1]  # Y is inverted in screen coords

            return proj_x, proj_y
        except Exception as e:
            debug_print(f"Error in project_3d_to_2d: {str(e)}")
            return 0, 0  # Return origin on error

    def setStructureData(self, nodes, members, supports):
        """Update the structure data"""
        try:
            debug_print(f"Setting structure data: {len(nodes)} nodes, {len(members)} members")
            self.nodes = nodes
            self.members = members
            self.supports = supports

            # Recenter the view on the structure
            self.recenter_view()

            # Update the scene with the new data
            self.update_scene()
        except Exception as e:
            debug_print(f"Error in setStructureData: {str(e)}")
            traceback.print_exc()

    def recenter_view(self):
        """Center the view on the structure"""
        try:
            debug_print("Recentering view")
            if not self.nodes:
                debug_print("No nodes to center on")
                return

            # Find structure bounds
            x_vals = []
            y_vals = []
            z_vals = []

            for pos in self.nodes.values():
                if len(pos) == 3:
                    x, y, z = pos
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)

            if not x_vals or not y_vals or not z_vals:
                debug_print("No valid coordinates to center on")
                return

            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            z_min, z_max = min(z_vals), max(z_vals)

            # Calculate center
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            # Set center offset to center the structure
            # We'll use the projection to determine the offset
            center_x, center_y = self.project_3d_to_2d(x_center, y_center, z_center)

            # Determine the view center
            view_center_x = self.width() / 2
            view_center_y = self.height() / 2

            # Calculate offset to center the structure
            self.center_offset = (
                view_center_x - center_x + 500,  # Add 500 because that's our base offset
                view_center_y - center_y + 300   # Add 300 because that's our base offset
            )

            # Reset the view
            self.resetTransform()
            self.centerOn(view_center_x, view_center_y)

            debug_print(f"View centered. Center offset: {self.center_offset}")
        except Exception as e:
            debug_print(f"Error in recenter_view: {str(e)}")
            traceback.print_exc()

    def update_scene(self):
        """Update the scene with current structure data"""
        debug_print("Updating scene")
        try:
            # Clear previous scene
            self.scene.clear()

            # Draw grid
            self.draw_grid()

            # Draw coordinate axes
            if self.show_axes:
                self.draw_axes()

            # Draw members first (so nodes are on top)
            self.draw_members()

            # Draw nodes
            self.draw_nodes()

            # Draw supports
            self.draw_supports()

            # Update the view
            self.update()
            debug_print("Scene updated successfully")
        except Exception as e:
            debug_print(f"Error updating scene: {str(e)}")
            traceback.print_exc()

    def draw_axes(self):
        """Draw X, Y, Z coordinate axes with clear right-handed system"""
        debug_print("Drawing coordinate axes")
        try:
            # Get origin position
            origin_x, origin_y = self.project_3d_to_2d(0, 0, 0)

            # Draw X axis (red) - points right
            x_end_x, x_end_y = self.project_3d_to_2d(10, 0, 0)
            x_pen = QPen(QColor(255, 0, 0))  # Red
            x_pen.setWidth(2)
            self.scene.addLine(origin_x, origin_y, x_end_x, x_end_y, x_pen)
            self.scene.addText("X").setPos(x_end_x + 5, x_end_y - 10)

            # Draw Y axis (green) - points forward/into screen
            y_end_x, y_end_y = self.project_3d_to_2d(0, 10, 0)
            y_pen = QPen(QColor(0, 180, 0))  # Green
            y_pen.setWidth(2)
            self.scene.addLine(origin_x, origin_y, y_end_x, y_end_y, y_pen)
            self.scene.addText("Y").setPos(y_end_x + 5, y_end_y - 10)

            # Draw Z axis (blue) - points up
            z_end_x, z_end_y = self.project_3d_to_2d(0, 0, 10)
            z_pen = QPen(QColor(0, 0, 255))  # Blue
            z_pen.setWidth(2)
            self.scene.addLine(origin_x, origin_y, z_end_x, z_end_y, z_pen)
            self.scene.addText("Z").setPos(z_end_x + 5, z_end_y - 10)

            # Add a small circle at origin
            self.scene.addEllipse(origin_x - 3, origin_y - 3, 6, 6,
                                 QPen(QColor(0, 0, 0)), QBrush(QColor(0, 0, 0)))

            # Draw the right-hand rule illustration
            # Draw a small curved arrow showing the rotation from X to Y
            arrow_path = QPainterPath()
            arrow_path.moveTo(origin_x + 15, origin_y - 5)
            arrow_path.arcTo(origin_x - 10, origin_y - 15, 30, 30, 0, -90)
            arrow_pen = QPen(QColor(100, 100, 100))
            arrow_pen.setWidth(1)
            self.scene.addPath(arrow_path, arrow_pen)

            # Add text explaining the right-hand rule
            self.scene.addText("Right-handed system: rotate X→Y, thumb points to Z").setPos(origin_x - 100, origin_y + 20)

        except Exception as e:
            debug_print(f"Error drawing axes: {str(e)}")
            traceback.print_exc()

    def draw_grid(self):
        """Draw a reference grid"""
        debug_print("Drawing grid")
        try:
            # Create a light gray pen
            grid_pen = QPen(QColor(200, 200, 200))
            grid_pen.setWidth(1)

            # Draw grid lines
            grid_size = 30
            step = 5

            for i in range(-grid_size, grid_size + 1, step):
                # Get projected coordinates
                start_x, start_y = self.project_3d_to_2d(i, -grid_size, 0)
                end_x, end_y = self.project_3d_to_2d(i, grid_size, 0)
                self.scene.addLine(start_x, start_y, end_x, end_y, grid_pen)

                start_x, start_y = self.project_3d_to_2d(-grid_size, i, 0)
                end_x, end_y = self.project_3d_to_2d(grid_size, i, 0)
                self.scene.addLine(start_x, start_y, end_x, end_y, grid_pen)
        except Exception as e:
            debug_print(f"Error drawing grid: {str(e)}")

    def draw_nodes(self):
        """Draw all nodes"""
        debug_print("Drawing nodes")
        try:
            for node_id, coords in self.nodes.items():
                try:
                    if len(coords) != 3:
                        debug_print(f"Invalid coordinates for node {node_id}: {coords}")
                        continue

                    x, y, z = coords

                    # Validate coordinates
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
                        debug_print(f"Non-numeric coordinates for node {node_id}: {coords}")
                        continue

                    # Project the 3D coordinates to 2D
                    proj_x, proj_y = self.project_3d_to_2d(x, y, z)

                    # Determine size and color based on selection
                    if node_id == self.selected_node:
                        size = 10
                        color = QColor(255, 0, 0)  # Red for selected
                    else:
                        size = 6
                        color = QColor(0, 0, 0)    # Black for normal

                    # Create node representation
                    node_item = self.scene.addEllipse(
                        proj_x - size/2, proj_y - size/2, size, size,
                        QPen(color), QBrush(color)
                    )

                    # Add node ID text
                    text_item = self.scene.addText(str(node_id))
                    text_item.setPos(proj_x + 5, proj_y - 15)
                    text_item.setDefaultTextColor(QColor(0, 0, 0))
                except Exception as e:
                    debug_print(f"Error drawing node {node_id}: {str(e)}")
                    continue
        except Exception as e:
            debug_print(f"Error in draw_nodes: {str(e)}")
            traceback.print_exc()

    def draw_members(self):
        """Draw all members"""
        debug_print("Drawing members")
        try:
            for member in self.members:
                try:
                    member_id = member.get('id')
                    i_node = member.get('i_node')
                    j_node = member.get('j_node')
                    section = member.get('section')

                    if any(param is None for param in [member_id, i_node, j_node, section]):
                        debug_print(f"Incomplete member data: {member}")
                        continue

                    # Skip if nodes don't exist
                    if i_node not in self.nodes or j_node not in self.nodes:
                        debug_print(f"Member {member_id} references non-existent node(s)")
                        continue

                    # Get node positions
                    coords1 = self.nodes[i_node]
                    coords2 = self.nodes[j_node]

                    if len(coords1) != 3 or len(coords2) != 3:
                        debug_print(f"Invalid node coordinates for member {member_id}")
                        continue

                    x1, y1, z1 = coords1
                    x2, y2, z2 = coords2

                    # Validate coordinates
                    if not all(isinstance(val, (int, float)) for val in [x1, y1, z1, x2, y2, z2]):
                        debug_print(f"Non-numeric coordinates for member {member_id}")
                        continue

                    # Project to 2D
                    p1x, p1y = self.project_3d_to_2d(x1, y1, z1)
                    p2x, p2y = self.project_3d_to_2d(x2, y2, z2)

                    # Determine line properties
                    if member_id == self.selected_member:
                        color = QColor(255, 0, 0)  # Red for selected
                        width = 3
                    else:
                        color = self.section_colors.get(section, QColor(0, 0, 0))
                        width = 2 if section == 1 else 1

                    # Create pen
                    pen = QPen(color)
                    pen.setWidth(width)

                    # Add line
                    line_item = self.scene.addLine(p1x, p1y, p2x, p2y, pen)

                    # Add member ID at midpoint
                    mid_x = (p1x + p2x) / 2
                    mid_y = (p1y + p2y) / 2

                    text_item = self.scene.addText(str(member_id))
                    text_item.setPos(mid_x, mid_y)
                    text_item.setDefaultTextColor(QColor(100, 100, 100))  # Gray text
                except Exception as e:
                    debug_print(f"Error drawing member {member.get('id')}: {str(e)}")
                    continue
        except Exception as e:
            debug_print(f"Error in draw_members: {str(e)}")
            traceback.print_exc()

    def draw_supports(self):
        """Draw support indicators"""
        debug_print("Drawing supports")
        try:
            for support in self.supports:
                try:
                    node_id = support.get('node')

                    if node_id is None:
                        debug_print(f"Support missing node reference: {support}")
                        continue

                    # Skip if node doesn't exist
                    if node_id not in self.nodes:
                        debug_print(f"Support references non-existent node: {node_id}")
                        continue

                    coords = self.nodes[node_id]
                    if len(coords) != 3:
                        debug_print(f"Invalid coordinates for node {node_id}: {coords}")
                        continue

                    # Get node position
                    x, y, z = coords

                    # Validate coordinates
                    if not all(isinstance(val, (int, float)) for val in [x, y, z]):
                        debug_print(f"Non-numeric coordinates for node {node_id}")
                        continue

                    # Project to 2D
                    proj_x, proj_y = self.project_3d_to_2d(x, y, z)

                    # Determine support type
                    is_fixed = support.get('ux') == 1 and support.get('uy') == 1 and support.get('uz') == 1

                    if is_fixed:
                        # Draw triangle for fixed support
                        triangle_path = QPainterPath()
                        triangle_path.moveTo(proj_x, proj_y + 10)
                        triangle_path.lineTo(proj_x - 10, proj_y + 20)
                        triangle_path.lineTo(proj_x + 10, proj_y + 20)
                        triangle_path.closeSubpath()

                        self.scene.addPath(triangle_path, QPen(QColor(255, 0, 0)), QBrush(QColor(255, 0, 0)))
                    else:
                        # Draw circle for other support
                        self.scene.addEllipse(
                            proj_x - 6, proj_y + 10, 12, 12,
                            QPen(QColor(255, 100, 0)), QBrush(QColor(255, 100, 0))
                        )
                except Exception as e:
                    debug_print(f"Error drawing support for node {support.get('node')}: {str(e)}")
                    continue
        except Exception as e:
            debug_print(f"Error in draw_supports: {str(e)}")
            traceback.print_exc()

    def resetView(self):
        """Reset the view to default position"""
        debug_print("Resetting view")
        try:
            self.resetTransform()
            self.recenter_view()
            self.update_scene()
        except Exception as e:
            debug_print(f"Error in resetView: {str(e)}")
            traceback.print_exc()


class StairwayInteractiveEditor(QMainWindow):
    """Main application window"""

    def __init__(self, json_file):
        super().__init__()
        try:
            debug_print(f"Initializing editor with file: {json_file}")
            self.json_file = json_file
            self.stairway_data = None
            self.current_mode = "view"

            # Create the structure view
            self.structure_view = StructureView()

            # Initialize data
            self.load_data()

            # Set up UI
            self.init_ui()

            # Update view with structure data
            debug_print("Setting initial structure data to view")
            self.structure_view.setStructureData(self.node_dict, self.members, self.supports)

            # Connect signals
            self.structure_view.nodeSelected.connect(self.on_node_selected)
            self.structure_view.memberSelected.connect(self.on_member_selected)

            # Setup mode state for member editing
            self.member_creation_nodes = []

            debug_print("Editor initialization complete")
        except Exception as e:
            debug_print(f"Error initializing editor: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "Initialization Error",
                               f"Failed to initialize application: {str(e)}")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Stairway Structure Editor - {os.path.basename(self.json_file)}")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center - structure view (already created in __init__)
        main_layout.addWidget(self.structure_view, 4)

        # Right panel - properties
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        """Create the left control panel"""
        debug_print("Creating left panel")
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Mode selection
        mode_group = QGroupBox("Edit Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.mode_buttons = QButtonGroup()
        modes = ["View", "Move Node", "Add Node", "Add Member", "Delete Node", "Delete Member", "Edit Member"]
        for i, mode in enumerate(modes):
            radio = QRadioButton(mode)
            mode_layout.addWidget(radio)
            self.mode_buttons.addButton(radio, i)
            if i == 0:
                radio.setChecked(True)

        self.mode_buttons.buttonClicked.connect(self.on_mode_changed)
        layout.addWidget(mode_group)

        # Section selection for new members
        section_group = QGroupBox("Member Section")
        section_layout = QVBoxLayout()
        section_group.setLayout(section_layout)

        self.section_combo = QComboBox()
        self.section_combo.addItems([f"Section {i}" for i in range(1, 6)])
        section_layout.addWidget(self.section_combo)
        layout.addWidget(section_group)

        # Material selection
        material_group = QGroupBox("Member Material")
        material_layout = QVBoxLayout()
        material_group.setLayout(material_layout)

        self.material_combo = QComboBox()
        self.material_combo.addItems([f"Material {i}" for i in range(1, 4)])
        material_layout.addWidget(self.material_combo)
        layout.addWidget(material_group)

        # Action buttons
        self.save_button = QPushButton("Save Structure")
        self.save_button.clicked.connect(self.save_structure)
        layout.addWidget(self.save_button)

        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.structure_view.resetView)
        layout.addWidget(self.reset_view_button)

        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)

        self.show_nodes_cb = QCheckBox("Show Node IDs")
        self.show_nodes_cb.setChecked(True)
        self.show_nodes_cb.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_nodes_cb)

        self.show_members_cb = QCheckBox("Show Member IDs")
        self.show_members_cb.setChecked(True)
        self.show_members_cb.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_members_cb)

        # Add checkbox for coordinate axes
        self.show_axes_cb = QCheckBox("Show Coordinate Axes")
        self.show_axes_cb.setChecked(True)
        self.show_axes_cb.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_axes_cb)

        layout.addWidget(display_group)

        # Info text
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        layout.addWidget(self.info_text)

        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create the right properties panel"""
        debug_print("Creating right panel")
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Node properties
        self.node_group = QGroupBox("Node Properties")
        node_layout = QVBoxLayout()
        self.node_group.setLayout(node_layout)

        # Node ID
        node_id_layout = QHBoxLayout()
        node_id_layout.addWidget(QLabel("ID:"))
        self.node_id_label = QLabel("-")
        node_id_layout.addWidget(self.node_id_label)
        node_id_layout.addStretch()
        node_layout.addLayout(node_id_layout)

        # Node coordinates
        self.node_x_spin = QDoubleSpinBox()
        self.node_x_spin.setRange(-1000, 1000)
        self.node_x_spin.setSingleStep(0.1)
        self.node_x_spin.valueChanged.connect(self.update_node_position)

        self.node_y_spin = QDoubleSpinBox()
        self.node_y_spin.setRange(-1000, 1000)
        self.node_y_spin.setSingleStep(0.1)
        self.node_y_spin.valueChanged.connect(self.update_node_position)

        self.node_z_spin = QDoubleSpinBox()
        self.node_z_spin.setRange(-1000, 1000)
        self.node_z_spin.setSingleStep(0.1)
        self.node_z_spin.valueChanged.connect(self.update_node_position)

        coord_layout = QVBoxLayout()
        for label, spin in [("X:", self.node_x_spin), ("Y:", self.node_y_spin), ("Z:", self.node_z_spin)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(spin)
            coord_layout.addLayout(row)

        node_layout.addLayout(coord_layout)
        self.node_group.setEnabled(False)
        layout.addWidget(self.node_group)

        # Member properties
        self.member_group = QGroupBox("Member Properties")
        member_layout = QVBoxLayout()
        self.member_group.setLayout(member_layout)

        # Member ID
        member_id_layout = QHBoxLayout()
        member_id_layout.addWidget(QLabel("ID:"))
        self.member_id_label = QLabel("-")
        member_id_layout.addWidget(self.member_id_label)
        member_id_layout.addStretch()
        member_layout.addLayout(member_id_layout)

        # Member nodes
        nodes_layout = QHBoxLayout()
        nodes_layout.addWidget(QLabel("Nodes:"))
        self.member_nodes_label = QLabel("-")
        nodes_layout.addWidget(self.member_nodes_label)
        nodes_layout.addStretch()
        member_layout.addLayout(nodes_layout)

        # Member section
        self.member_section_combo = QComboBox()
        self.member_section_combo.addItems([f"Section {i}" for i in range(1, 6)])
        self.member_section_combo.currentIndexChanged.connect(self.update_member_properties)

        section_layout = QHBoxLayout()
        section_layout.addWidget(QLabel("Section:"))
        section_layout.addWidget(self.member_section_combo)
        member_layout.addLayout(section_layout)

        # Member material
        self.member_material_combo = QComboBox()
        self.member_material_combo.addItems([f"Material {i}" for i in range(1, 4)])
        self.member_material_combo.currentIndexChanged.connect(self.update_member_properties)

        material_layout = QHBoxLayout()
        material_layout.addWidget(QLabel("Material:"))
        material_layout.addWidget(self.member_material_combo)
        member_layout.addLayout(material_layout)

        # Change nodes button
        self.change_nodes_button = QPushButton("Change Connected Nodes")
        self.change_nodes_button.clicked.connect(self.start_change_member_nodes)
        member_layout.addWidget(self.change_nodes_button)

        self.member_group.setEnabled(False)
        layout.addWidget(self.member_group)

        # Structure tree
        tree_group = QGroupBox("Structure Tree")
        tree_layout = QVBoxLayout()
        tree_group.setLayout(tree_layout)

        self.structure_tree = QTreeWidget()
        self.structure_tree.setHeaderLabels(["Element", "Type", "Properties"])
        tree_layout.addWidget(self.structure_tree)
        self.populate_structure_tree()

        layout.addWidget(tree_group)

        layout.addStretch()

        return panel

    def populate_structure_tree(self):
        """Populate the structure tree with nodes and members"""
        debug_print("Populating structure tree")
        try:
            self.structure_tree.clear()

            # Add nodes
            nodes_item = QTreeWidgetItem(["Nodes", f"{len(self.nodes)}", ""])
            self.structure_tree.addTopLevelItem(nodes_item)

            # Add members grouped by section
            sections = {}
            for member in self.members:
                section = member.get('section', 0)
                if section not in sections:
                    sections[section] = []
                sections[section].append(member)

            for section, members in sections.items():
                section_item = QTreeWidgetItem([f"Section {section}", f"{len(members)} members", ""])
                self.structure_tree.addTopLevelItem(section_item)

            self.structure_tree.expandAll()
        except Exception as e:
            debug_print(f"Error populating structure tree: {str(e)}")

    def load_data(self):
        """Load structure data from JSON file"""
        debug_print(f"Loading structure from {self.json_file}")

        try:
            with open(self.json_file, 'r') as file:
                json_str = ""
                for line in file:
                    if '//' not in line:
                        json_str += line

                self.stairway_data = json.loads(json_str)

            # Extract data
            self.nodes = self.stairway_data.get('nodes', [])
            self.supports = self.stairway_data.get('supports', [])
            self.members = self.stairway_data.get('members', [])
            self.materials = self.stairway_data.get('materials', [])
            self.sections = self.stairway_data.get('sections', [])

            # Create node dictionary
            self.node_dict = {}
            for node in self.nodes:
                try:
                    node_id = node.get('id')
                    if node_id is None:
                        continue

                    # Extract and convert coordinates
                    x_str = node.get('x', '0 ft')
                    y_str = node.get('y', '0 ft')
                    z_str = node.get('z', '0 ft')

                    # Handle different format strings
                    try:
                        x = float(x_str.split()[0])
                        y = float(y_str.split()[0])
                        z = float(z_str.split()[0])
                    except (ValueError, IndexError):
                        debug_print(f"Error parsing coordinates for node {node_id}: {x_str}, {y_str}, {z_str}")
                        continue

                    self.node_dict[node_id] = (x, y, z)
                except Exception as e:
                    debug_print(f"Error processing node: {str(e)}")
                    continue

            # Create connections map
            self.node_connections = {node_id: [] for node_id in self.node_dict}
            for member in self.members:
                try:
                    i_node = member.get('i_node')
                    j_node = member.get('j_node')
                    member_id = member.get('id')

                    if None in (i_node, j_node, member_id):
                        debug_print(f"Incomplete member data: {member}")
                        continue

                    if i_node in self.node_connections:
                        self.node_connections[i_node].append(member_id)
                    if j_node in self.node_connections:
                        self.node_connections[j_node].append(member_id)
                except Exception as e:
                    debug_print(f"Error processing member connection: {str(e)}")
                    continue

            debug_print(f"Loaded {len(self.nodes)} nodes, {len(self.members)} members")

        except Exception as e:
            debug_print(f"Error loading file: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            sys.exit(1)

    def update_display_options(self):
        """Update display options"""
        try:
            debug_print("Updating display options")
            # Update the display options in the view
            self.structure_view.show_axes = self.show_axes_cb.isChecked()
            self.structure_view.update_scene()
        except Exception as e:
            debug_print(f"Error updating display options: {str(e)}")

    def on_mode_changed(self, button):
        """Handle mode change"""
        try:
            mode_index = self.mode_buttons.id(button)
            modes = ["view", "move", "add_node", "add_member", "delete_node", "delete_member", "edit_member"]
            self.current_mode = modes[mode_index]
            debug_print(f"Mode changed to: {self.current_mode}")

            # Reset state when changing modes
            if self.current_mode == "add_member":
                self.member_creation_nodes = []

            # Update UI based on mode
            if self.current_mode == "move":
                self.node_group.setEnabled(self.structure_view.selected_node is not None)
                self.member_group.setEnabled(False)
            elif self.current_mode == "edit_member":
                self.node_group.setEnabled(False)
                self.member_group.setEnabled(self.structure_view.selected_member is not None)
            else:
                self.node_group.setEnabled(False)
                self.member_group.setEnabled(False)

            self.update_info_text()
        except Exception as e:
            debug_print(f"Error changing mode: {str(e)}")

    def on_node_selected(self, node_id):
        """Handle node selection"""
        try:
            debug_print(f"Node {node_id} selected")

            # Handle node selection for add_member mode
            if self.current_mode == "add_member":
                if node_id in self.member_creation_nodes:
                    debug_print(f"Node {node_id} already selected")
                    return

                self.member_creation_nodes.append(node_id)
                debug_print(f"Added node {node_id} to member creation (nodes: {self.member_creation_nodes})")

                if len(self.member_creation_nodes) == 2:
                    self.create_member(self.member_creation_nodes[0], self.member_creation_nodes[1])
                    self.member_creation_nodes = []

                self.update_info_text()
                return

            # Ensure node exists in dictionary
            if node_id not in self.node_dict:
                debug_print(f"Selected node {node_id} not found in node dictionary")
                return

            # Update node properties panel
            self.node_id_label.setText(str(node_id))
            x, y, z = self.node_dict[node_id]

            # Block signals to prevent recursive updates
            self.node_x_spin.blockSignals(True)
            self.node_y_spin.blockSignals(True)
            self.node_z_spin.blockSignals(True)

            self.node_x_spin.setValue(x)
            self.node_y_spin.setValue(y)
            self.node_z_spin.setValue(z)

            self.node_x_spin.blockSignals(False)
            self.node_y_spin.blockSignals(False)
            self.node_z_spin.blockSignals(False)

            # Enable/disable panels based on mode
            if self.current_mode == "move":
                self.node_group.setEnabled(True)
            elif self.current_mode == "delete_node":
                self.delete_node(node_id)

            self.member_group.setEnabled(False)
            self.update_info_text()
        except Exception as e:
            debug_print(f"Error in node selection: {str(e)}")

    def on_member_selected(self, member_id):
        """Handle member selection"""
        try:
            debug_print(f"Member {member_id} selected")

            if self.current_mode == "delete_member":
                self.delete_member(member_id)
                return

            # Find member data
            member = next((m for m in self.members if m.get('id') == member_id), None)
            if not member:
                debug_print(f"Member {member_id} not found in members list")
                return

            # Update member properties panel
            self.member_id_label.setText(str(member_id))
            i_node = member.get('i_node', '-')
            j_node = member.get('j_node', '-')
            self.member_nodes_label.setText(f"{i_node} - {j_node}")

            # Block signals
            self.member_section_combo.blockSignals(True)
            self.member_material_combo.blockSignals(True)

            # Update combo box selections
            section = member.get('section', 1)
            material = member.get('material', 1)
            self.member_section_combo.setCurrentIndex(section - 1)
            self.member_material_combo.setCurrentIndex(material - 1)

            self.member_section_combo.blockSignals(False)
            self.member_material_combo.blockSignals(False)

            # Enable/disable panels based on mode
            self.node_group.setEnabled(False)
            if self.current_mode == "edit_member":
                self.member_group.setEnabled(True)

            self.update_info_text()
        except Exception as e:
            debug_print(f"Error in member selection: {str(e)}")

    def update_node_position(self):
        """Update selected node position from spinboxes"""
        try:
            if not self.structure_view.selected_node:
                return

            node_id = self.structure_view.selected_node
            x = self.node_x_spin.value()
            y = self.node_y_spin.value()
            z = self.node_z_spin.value()

            # Update internal data
            self.node_dict[node_id] = (x, y, z)

            # Update node in list
            for node in self.nodes:
                if node.get('id') == node_id:
                    node['x'] = f"{x} ft"
                    node['y'] = f"{y} ft"
                    node['z'] = f"{z} ft"
                    break

            # Update view
            self.structure_view.setStructureData(self.node_dict, self.members, self.supports)
        except Exception as e:
            debug_print(f"Error updating node position: {str(e)}")

    def update_member_properties(self):
        """Update selected member properties"""
        try:
            if not self.structure_view.selected_member:
                return

            member_id = self.structure_view.selected_member

            # Find member
            for member in self.members:
                if member.get('id') == member_id:
                    member['section'] = self.member_section_combo.currentIndex() + 1
                    member['material'] = self.member_material_combo.currentIndex() + 1
                    break

            # Update view
            self.structure_view.update_scene()
        except Exception as e:
            debug_print(f"Error updating member properties: {str(e)}")

    def start_change_member_nodes(self):
        """Start process to change member's connected nodes"""
        try:
            if not self.structure_view.selected_member:
                return

            # Switch to special mode for changing member nodes
            self.current_mode = "change_member_nodes"
            self.member_creation_nodes = []

            QMessageBox.information(self, "Change Nodes",
                                  "Select two nodes to reconnect the member")
            self.update_info_text()
        except Exception as e:
            debug_print(f"Error starting member node change: {str(e)}")

    def update_member_nodes(self, member_id, new_i_node, new_j_node):
        """Update a member's connected nodes"""
        try:
            member = next((m for m in self.members if m.get('id') == member_id), None)
            if not member:
                return

            old_i_node = member.get('i_node')
            old_j_node = member.get('j_node')

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

            # Update view
            self.structure_view.setStructureData(self.node_dict, self.members, self.supports)
        except Exception as e:
            debug_print(f"Error updating member nodes: {str(e)}")

    def create_member(self, node1, node2):
        """Create a new member between two nodes"""
        try:
            section = self.section_combo.currentIndex() + 1
            material = self.material_combo.currentIndex() + 1

            # Get next member ID
            member_ids = [m.get('id', 0) for m in self.members]
            next_id = max(member_ids, default=0) + 1

            # Create new member
            new_member = {
                'id': next_id,
                'i_node': node1,
                'j_node': node2,
                'section': section,
                'material': material
            }

            debug_print(f"Creating member {next_id} between nodes {node1}-{node2}")

            # Add to data structures
            self.members.append(new_member)

            # Update connections
            if node1 not in self.node_connections:
                self.node_connections[node1] = []
            self.node_connections[node1].append(next_id)

            if node2 not in self.node_connections:
                self.node_connections[node2] = []
            self.node_connections[node2].append(next_id)

            # Update view
            self.structure_view.setStructureData(self.node_dict, self.members, self.supports)
            self.populate_structure_tree()
        except Exception as e:
            debug_print(f"Error creating member: {str(e)}")

    def delete_node(self, node_id):
        """Delete a node and its connected members"""
        try:
            debug_print(f"Deleting node {node_id}")

            # Get connected members
            connected_members = self.node_connections.get(node_id, [])[:]

            # Delete connected members
            if connected_members:
                for member_id in connected_members:
                    self.delete_member(member_id, update_viz=False)

            # Delete node
            self.nodes = [n for n in self.nodes if n.get('id') != node_id]
            if node_id in self.node_dict:
                del self.node_dict[node_id]
            if node_id in self.node_connections:
                del self.node_connections[node_id]

            # Update view
            self.structure_view.selected_node = None
            self.structure_view.setStructureData(self.node_dict, self.members, self.supports)
            self.populate_structure_tree()
            self.update_info_text()
        except Exception as e:
            debug_print(f"Error deleting node: {str(e)}")

    def delete_member(self, member_id, update_viz=True):
        """Delete a member"""
        try:
            debug_print(f"Deleting member {member_id}")

            # Find member
            member = next((m for m in self.members if m.get('id') == member_id), None)
            if not member:
                return

            # Remove from connections
            i_node = member.get('i_node')
            j_node = member.get('j_node')

            if i_node in self.node_connections and member_id in self.node_connections[i_node]:
                self.node_connections[i_node].remove(member_id)

            if j_node in self.node_connections and member_id in self.node_connections[j_node]:
                self.node_connections[j_node].remove(member_id)

            # Remove from list
            self.members = [m for m in self.members if m.get('id') != member_id]

            # Update visualization if requested
            if update_viz:
                self.structure_view.selected_member = None
                self.structure_view.setStructureData(self.node_dict, self.members, self.supports)
                self.populate_structure_tree()
                self.update_info_text()
        except Exception as e:
            debug_print(f"Error deleting member: {str(e)}")

    def update_info_text(self):
        """Update information text"""
        try:
            info = f"Mode: {self.current_mode.replace('_', ' ').title()}\n\n"

            if self.current_mode == "add_member":
                info += f"Select nodes to create a member.\n"
                info += f"Selected nodes: {len(self.member_creation_nodes)}/2\n\n"
            elif self.current_mode == "change_member_nodes":
                info += f"Select nodes to reconnect member {self.structure_view.selected_member}.\n"
                info += f"Selected nodes: {len(self.member_creation_nodes)}/2\n\n"

            if self.structure_view.selected_node:
                node_id = self.structure_view.selected_node
                if node_id in self.node_dict:
                    x, y, z = self.node_dict[node_id]
                    info += f"Selected Node: {node_id}\n"
                    info += f"Position: ({x:.2f}, {y:.2f}, {z:.2f})\n"
                    info += f"Connected Members: {len(self.node_connections.get(node_id, []))}\n"

            elif self.structure_view.selected_member:
                member = next((m for m in self.members if m.get('id') == self.structure_view.selected_member), None)
                if member:
                    info += f"Selected Member: {member.get('id')}\n"
                    info += f"Nodes: {member.get('i_node')} - {member.get('j_node')}\n"
                    info += f"Section: {member.get('section')}, Material: {member.get('material')}\n"

                    # Calculate length if both nodes exist
                    i_node = member.get('i_node')
                    j_node = member.get('j_node')
                    if i_node in self.node_dict and j_node in self.node_dict:
                        i_pos = self.node_dict[i_node]
                        j_pos = self.node_dict[j_node]
                        # Use math.sqrt instead of sp.sqrt
                        length = math.sqrt(sum((j_pos[i] - i_pos[i])**2 for i in range(3)))
                        info += f"Length: {length:.2f} ft\n"

            self.info_text.setText(info)
        except Exception as e:
            debug_print(f"Error updating info text: {str(e)}")

    def save_structure(self):
        """Save structure to JSON file"""
        try:
            # Update structure data
            self.stairway_data['nodes'] = self.nodes
            self.stairway_data['members'] = self.members

            # Generate filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(self.json_file)[0]
            output_file = f"{base_name}_edited_{timestamp}.json"

            try:
                with open(output_file, 'w') as f:
                    json.dump(self.stairway_data, f, indent=4)
                QMessageBox.information(self, "Success", f"Structure saved to {output_file}")
                debug_print(f"Saved structure to {output_file}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
                debug_print(f"Error saving: {str(e)}")
        except Exception as e:
            debug_print(f"Error in save_structure: {str(e)}")


def main():
    """Main entry point"""
    try:
        # Direct console output to stderr for debugging
        sys.stdout.flush()

        debug_print(f"Starting application with Python {sys.version}")

        # Create QApplication
        app = QApplication(sys.argv)

        # Get JSON file
        if len(sys.argv) > 1:
            json_file = sys.argv[1]
        else:
            json_file = "stairway_structure_Imperial.JSON"
            if not os.path.exists(json_file):
                # Try to find any JSON file
                json_files = [f for f in os.listdir('.') if f.endswith('.json')]
                if json_files:
                    json_file = json_files[0]
                else:
                    debug_print("No JSON files found")
                    QMessageBox.critical(None, "Error", "No JSON files found in the current directory")
                    sys.exit(1)

        # Create and show main window with exception handling
        editor = StairwayInteractiveEditor(json_file)
        editor.show()
        debug_print("Application started successfully")

        # Set up a timer to periodically flush stdout
        flush_timer = QTimer()
        flush_timer.timeout.connect(lambda: sys.stdout.flush())
        flush_timer.start(1000)  # Flush every second

        sys.exit(app.exec_())
    except Exception as e:
        debug_print(f"Fatal error in main: {str(e)}")
        traceback.print_exc()

        # Try to show error message to user
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("Fatal Error")
            error_msg.setText("The application has encountered a critical error and needs to close.")
            error_msg.setDetailedText(f"Error: {str(e)}\n\n{traceback.format_exc()}")
            error_msg.exec_()
        except:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
