# Interactive stairway structure visualization and editing tool using PyQt5 and OpenGL

import os
import json
import sys
import datetime
import scipy as sp
from scipy.spatial import distance
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QSlider,
                            QGroupBox, QRadioButton, QButtonGroup, QSplitter,
                            QTextEdit, QFileDialog, QMessageBox, QSpinBox,
                            QDoubleSpinBox, QComboBox, QTreeWidget, QTreeWidgetItem,
                            QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QVector3D, QMatrix4x4, QQuaternion, QPainter, QFont, QColor
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# Global debug flag
DEBUG = True

def debug_print(*args, **kwargs):
    """Helper function for debug printing"""
    if DEBUG:
        print(*args, **kwargs)

class StructureGLWidget(QGLWidget):
    """OpenGL widget for 3D structure visualization"""

    nodeSelected = pyqtSignal(int)
    memberSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Enable tracking mouse movements (even without button press)
        self.setMouseTracking(True)

        # Camera parameters
        self.camera_distance = 50.0
        self.camera_rotation = [20.0, -60.0]  # elevation, azimuth
        self.camera_target = [0.0, 0.0, 0.0]

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5

        # Structure data
        self.nodes = {}
        self.members = []
        self.supports = []

        # Selection
        self.selected_node = None
        self.selected_member = None
        self.hover_node = None
        self.hover_member = None

        # Display options
        self.show_node_ids = True
        self.show_member_ids = False
        self.show_grid = True
        self.show_axes_labels = True

        # Colors for sections
        self.section_colors = {
            1: (0.0, 0.0, 1.0),    # Blue - Main stringers
            2: (0.0, 1.0, 0.0),    # Green - Steps/treads
            3: (0.0, 1.0, 1.0),    # Cyan - Secondary members
            4: (0.5, 0.5, 0.5),    # Gray - Bracing
            5: (1.0, 0.0, 0.0)     # Red - Column
        }

        # Lists for text rendering (using QPainter instead of renderText)
        self.node_labels = []  # Stores (x, y, text) for node labels
        self.member_labels = []  # Stores (x, y, text) for member labels
        self.axes_labels = []  # Stores (x, y, text) for axis labels

    def initializeGL(self):
        """Initialize OpenGL settings"""
        debug_print("Initializing OpenGL...")

        glClearColor(0.95, 0.95, 0.95, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Disable lighting initially - we'll use simple colors
        glDisable(GL_LIGHTING)

        # Set up right-handed coordinate system by default in OpenGL
        # OpenGL uses right-handed by default, no change needed

        debug_print("OpenGL initialized successfully")

    def resizeGL(self, width, height):
        """Handle widget resize"""
        debug_print(f"Resizing GL widget to {width}x{height}")
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Render the scene"""
        # Clear lists for text rendering
        self.node_labels = []
        self.member_labels = []

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        eye_x = self.camera_target[0] + self.camera_distance * sp.cos(sp.radians(self.camera_rotation[0])) * sp.cos(sp.radians(self.camera_rotation[1]))
        eye_y = self.camera_target[1] + self.camera_distance * sp.cos(sp.radians(self.camera_rotation[0])) * sp.sin(sp.radians(self.camera_rotation[1]))
        eye_z = self.camera_target[2] + self.camera_distance * sp.sin(sp.radians(self.camera_rotation[0]))

        try:
            gluLookAt(eye_x, eye_y, eye_z,
                    self.camera_target[0], self.camera_target[1], self.camera_target[2],
                    0.0, 0.0, 1.0)

            # Draw grid
            if self.show_grid:
                self.drawGrid()

            # Draw structure
            self.drawMembers()
            self.drawNodes()
            self.drawSupports()

            # Draw axes
            self.drawAxes()

        except Exception as e:
            debug_print(f"Error in paintGL: {e}")

    def paintEvent(self, event):
        """Override to combine OpenGL rendering with QPainter for text"""
        # First do the standard OpenGL rendering
        self.makeCurrent()
        super().paintEvent(event)

        # Then overlay text with QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # Draw node labels
        if self.show_node_ids:
            painter.setPen(Qt.black)
            for x, y, text in self.node_labels:
                painter.drawText(int(x), int(y), text)

        # Draw member labels
        if self.show_member_ids:
            painter.setPen(Qt.darkGray)
            for x, y, text in self.member_labels:
                painter.drawText(int(x), int(y), text)

        # Draw axis labels
        if self.show_axes_labels:
            painter.setPen(Qt.black)
            for x, y, text in self.axes_labels:
                if text == "Right-handed system":
                    font = painter.font()
                    font.setPointSize(8)
                    painter.setFont(font)
                    painter.drawText(int(x), int(y), text)
                    font.setPointSize(10)
                    painter.setFont(font)
                else:
                    painter.drawText(int(x), int(y), text)

        painter.end()

    def drawGrid(self):
        """Draw reference grid"""
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(1.0)

        glBegin(GL_LINES)
        grid_size = 50
        grid_step = 5
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, -grid_size, 0)
            glVertex3f(i, grid_size, 0)
            glVertex3f(-grid_size, i, 0)
            glVertex3f(grid_size, i, 0)
        glEnd()

    def drawAxes(self):
        """Draw coordinate axes with right-hand rule indicator"""
        glLineWidth(2.0)

        # X axis - red
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(10, 0, 0)
        glEnd()

        # Y axis - green
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 10, 0)
        glEnd()

        # Z axis - blue
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 10)
        glEnd()

        # Add a small circular arrow indicating the right-hand rule from X to Y
        # Small circle in XY plane
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.5)

        radius = 3.0
        segments = 20
        angle_step = 2 * sp.pi / segments

        # Draw 3/4 of a circle
        glBegin(GL_LINE_STRIP)
        for i in range(int(segments * 0.75)):
            angle = i * angle_step
            glVertex3f(radius * sp.cos(angle), radius * sp.sin(angle), 0.1)
        glEnd()

        # Add arrowhead
        arrowhead_angle = sp.pi * 3 / 2
        arrowhead_length = 0.8
        glBegin(GL_TRIANGLES)
        glVertex3f(radius * sp.cos(arrowhead_angle), radius * sp.sin(arrowhead_angle), 0.1)
        glVertex3f(radius * sp.cos(arrowhead_angle) - arrowhead_length, radius * sp.sin(arrowhead_angle) + arrowhead_length, 0.1)
        glVertex3f(radius * sp.cos(arrowhead_angle) + arrowhead_length, radius * sp.sin(arrowhead_angle) + arrowhead_length, 0.1)
        glEnd()

        # Store axis labels for later rendering with QPainter
        if self.show_axes_labels:
            # Project axis endpoints to screen
            x_end = self.project3DToScreen(10, 0, 0)
            y_end = self.project3DToScreen(0, 10, 0)
            z_end = self.project3DToScreen(0, 0, 10)
            origin = self.project3DToScreen(0, 0, 0)

            if x_end and y_end and z_end and origin:
                self.axes_labels.append((x_end[0] + 5, x_end[1], "X"))
                self.axes_labels.append((y_end[0] + 5, y_end[1], "Y"))
                self.axes_labels.append((z_end[0] + 5, z_end[1], "Z"))
                self.axes_labels.append((origin[0] + 10, origin[1] + 15, "Right-handed system"))

    def drawNodes(self):
        """Draw all nodes"""
        for node_id, (x, y, z) in self.nodes.items():
            # Set color based on selection/hover state
            if node_id == self.selected_node:
                glColor3f(1.0, 0.0, 0.0)
                size = 10.0
            elif node_id == self.hover_node:
                glColor3f(1.0, 0.5, 0.0)
                size = 8.0
            else:
                glColor3f(0.0, 0.0, 0.0)
                size = 6.0

            # Draw node as point
            glPointSize(size)
            glBegin(GL_POINTS)
            glVertex3f(x, y, z)
            glEnd()

            # Add node ID to the list for later text rendering
            if self.show_node_ids:
                screen_pos = self.project3DToScreen(x, y, z)
                if screen_pos:
                    self.node_labels.append((screen_pos[0], screen_pos[1], str(node_id)))

    def drawMembers(self):
        """Draw all members"""
        for member in self.members:
            try:
                member_id = member['id']
                i_node = member['i_node']
                j_node = member['j_node']
                section = member['section']

                if i_node not in self.nodes or j_node not in self.nodes:
                    continue

                x1, y1, z1 = self.nodes[i_node]
                x2, y2, z2 = self.nodes[j_node]

                # Set color and width based on section and selection
                if member_id == self.selected_member:
                    glColor3f(1.0, 0.0, 0.0)
                    line_width = 4.0
                elif member_id == self.hover_member:
                    glColor3f(1.0, 0.5, 0.0)
                    line_width = 3.0
                else:
                    color = self.section_colors.get(section, (0.0, 0.0, 0.0))
                    glColor3f(*color)
                    line_width = 2.0 if section == 1 else 1.5

                # Apply line width
                glLineWidth(line_width)

                # Draw member as line
                glBegin(GL_LINES)
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
                glEnd()

                # Add member ID to the list for later text rendering
                if self.show_member_ids:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    mid_z = (z1 + z2) / 2
                    screen_pos = self.project3DToScreen(mid_x, mid_y, mid_z)
                    if screen_pos:
                        self.member_labels.append((screen_pos[0], screen_pos[1], str(member_id)))
            except Exception as e:
                debug_print(f"Error drawing member {member.get('id', 'unknown')}: {e}")

    def drawSupports(self):
        """Draw support constraints"""
        for support in self.supports:
            try:
                node_id = support['node']
                if node_id not in self.nodes:
                    continue

                x, y, z = self.nodes[node_id]

                # Full support (fixed)
                if support['ux'] == 1 and support['uy'] == 1 and support['uz'] == 1:
                    glColor3f(1.0, 0.0, 0.0)
                    # Draw as pyramid (simple triangle)
                    glBegin(GL_TRIANGLES)
                    glVertex3f(x - 0.5, y - 0.5, z - 0.5)
                    glVertex3f(x + 0.5, y - 0.5, z - 0.5)
                    glVertex3f(x, y + 0.5, z - 0.5)
                    glEnd()
                else:
                    # Partial support
                    glColor3f(1.0, 0.5, 0.0)
                    # Draw as simple square
                    glBegin(GL_QUADS)
                    glVertex3f(x - 0.5, y - 0.5, z - 0.5)
                    glVertex3f(x + 0.5, y - 0.5, z - 0.5)
                    glVertex3f(x + 0.5, y + 0.5, z - 0.5)
                    glVertex3f(x - 0.5, y + 0.5, z - 0.5)
                    glEnd()
            except Exception as e:
                debug_print(f"Error drawing support for node {support.get('node', 'unknown')}: {e}")

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.LeftButton and event.modifiers() != Qt.ControlModifier:
            # Selection
            node_id = self.pickNode(event.x(), event.y())
            if node_id is not None:
                self.selected_node = node_id
                self.selected_member = None
                self.nodeSelected.emit(node_id)
                debug_print(f"Selected node: {node_id}")
            else:
                member_id = self.pickMember(event.x(), event.y())
                if member_id is not None:
                    self.selected_member = member_id
                    self.selected_node = None
                    self.memberSelected.emit(member_id)
                    debug_print(f"Selected member: {member_id}")
                else:
                    self.selected_node = None
                    self.selected_member = None

            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if event.buttons() & Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            # Rotate camera
            self.camera_rotation[1] += dx * self.mouse_sensitivity
            self.camera_rotation[0] += dy * self.mouse_sensitivity
            self.camera_rotation[0] = max(-89, min(89, self.camera_rotation[0]))
            self.update()
        elif event.buttons() & Qt.MiddleButton:
            # Pan camera
            pan_speed = 0.1
            self.camera_target[0] -= dx * pan_speed
            self.camera_target[1] += dy * pan_speed
            self.update()

        self.last_mouse_pos = event.pos()

    def wheelEvent(self, event):
        """Handle mouse wheel events"""
        delta = event.angleDelta().y()
        zoom_speed = 0.1
        self.camera_distance *= (1.0 - delta / 120.0 * zoom_speed)
        self.camera_distance = max(5.0, min(200.0, self.camera_distance))
        self.update()

    def pickNode(self, x, y):
        """Find node at screen coordinates"""
        min_dist = float('inf')
        picked_node = None

        for node_id, (nx, ny, nz) in self.nodes.items():
            # Project 3D point to screen
            screen_pos = self.project3DToScreen(nx, ny, nz)
            if screen_pos is not None:
                sx, sy = screen_pos
                dist = sp.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < 20 and dist < min_dist:  # 20 pixel threshold
                    min_dist = dist
                    picked_node = node_id

        return picked_node

    def pickMember(self, x, y):
        """Find member at screen coordinates"""
        min_dist = float('inf')
        picked_member = None

        for member in self.members:
            i_node = member['i_node']
            j_node = member['j_node']

            if i_node not in self.nodes or j_node not in self.nodes:
                continue

            x1, y1, z1 = self.nodes[i_node]
            x2, y2, z2 = self.nodes[j_node]

            # Project endpoints to screen
            p1 = self.project3DToScreen(x1, y1, z1)
            p2 = self.project3DToScreen(x2, y2, z2)

            if p1 is not None and p2 is not None:
                # Calculate distance from point to line segment
                dist = self.pointToLineDistance(x, y, p1[0], p1[1], p2[0], p2[1])
                if dist < 10 and dist < min_dist:  # 10 pixel threshold
                    min_dist = dist
                    picked_member = member['id']

        return picked_member

    def project3DToScreen(self, x, y, z):
        """Project 3D point to screen coordinates"""
        try:
            # Get viewport
            viewport = glGetIntegerv(GL_VIEWPORT)

            # Get matrices
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)

            # Project
            result = gluProject(x, y, z, modelview, projection, viewport)
            if result[2] < 0 or result[2] > 1:  # Behind camera or too far
                return None

            return (result[0], viewport[3] - result[1])  # Flip Y coordinate
        except Exception as e:
            debug_print(f"Error projecting point: {e}")
            return None

    def pointToLineDistance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_length_sq == 0:
            return sp.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return sp.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def setStructureData(self, nodes, members, supports):
        """Update structure data"""
        debug_print(f"Setting structure data: {len(nodes)} nodes, {len(members)} members")
        self.nodes = nodes
        self.members = members
        self.supports = supports
        self.update()

    def resetView(self):
        """Reset camera to default view"""
        debug_print("Resetting view")
        self.camera_distance = 50.0
        self.camera_rotation = [20.0, -60.0]
        self.camera_target = [0.0, 0.0, 0.0]
        self.update()

class StairwayInteractiveEditor(QMainWindow):
    """Main application window"""

    def __init__(self, json_file):
        super().__init__()
        debug_print(f"Initializing editor with file: {json_file}")
        self.json_file = json_file
        self.stairway_data = None
        self.current_mode = "view"

        # Create GL widget first so it's available for connecting signals
        self.gl_widget = StructureGLWidget()

        # Initialize data
        self.load_data()

        # Set up UI
        self.init_ui()

        # Update GL widget with structure data
        debug_print("Setting initial structure data to GL widget")
        self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)

        # Connect signals
        self.gl_widget.nodeSelected.connect(self.on_node_selected)
        self.gl_widget.memberSelected.connect(self.on_member_selected)

        # Setup mode state for member editing
        self.member_creation_nodes = []

        debug_print("Editor initialization complete")

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

        # Center - GL widget (already created in __init__)
        main_layout.addWidget(self.gl_widget, 4)

        # Right panel - properties
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        """Create the left control panel"""
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
        self.reset_view_button.clicked.connect(self.reset_view)  # Now connects to our method
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
        self.show_members_cb.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_members_cb)

        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_grid_cb)

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
        self.structure_tree.clear()

        # Add nodes
        nodes_item = QTreeWidgetItem(["Nodes", f"{len(self.nodes)}", ""])
        self.structure_tree.addTopLevelItem(nodes_item)

        # Add members grouped by section
        sections = {}
        for member in self.members:
            section = member['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(member)

        for section, members in sections.items():
            section_item = QTreeWidgetItem([f"Section {section}", f"{len(members)} members", ""])
            self.structure_tree.addTopLevelItem(section_item)

        self.structure_tree.expandAll()

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
            self.nodes = self.stairway_data['nodes']
            self.supports = self.stairway_data['supports']
            self.members = self.stairway_data['members']
            self.materials = self.stairway_data.get('materials', [])
            self.sections = self.stairway_data.get('sections', [])

            # Create node dictionary
            self.node_dict = {}
            for node in self.nodes:
                node_id = node['id']
                x = float(node['x'].split()[0])
                y = float(node['y'].split()[0])
                z = float(node['z'].split()[0])
                self.node_dict[node_id] = (x, y, z)

            # Create connections map
            self.node_connections = {node_id: [] for node_id in self.node_dict}
            for member in self.members:
                i_node = member['i_node']
                j_node = member['j_node']
                member_id = member['id']

                if i_node in self.node_connections:
                    self.node_connections[i_node].append(member_id)
                if j_node in self.node_connections:
                    self.node_connections[j_node].append(member_id)

            debug_print(f"Loaded {len(self.nodes)} nodes, {len(self.members)} members")

        except Exception as e:
            debug_print(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            sys.exit(1)

    def update_display_options(self):
        """Update display options in GL widget"""
        self.gl_widget.show_node_ids = self.show_nodes_cb.isChecked()
        self.gl_widget.show_member_ids = self.show_members_cb.isChecked()
        self.gl_widget.show_grid = self.show_grid_cb.isChecked()
        self.gl_widget.update()

    def on_mode_changed(self, button):
        """Handle mode change"""
        mode_index = self.mode_buttons.id(button)
        modes = ["view", "move", "add_node", "add_member", "delete_node", "delete_member", "edit_member"]
        self.current_mode = modes[mode_index]
        debug_print(f"Mode changed to: {self.current_mode}")

        # Reset state when changing modes
        if self.current_mode == "add_member":
            self.member_creation_nodes = []

        # Update UI based on mode
        if self.current_mode == "move":
            self.node_group.setEnabled(self.gl_widget.selected_node is not None)
            self.member_group.setEnabled(False)
        elif self.current_mode == "edit_member":
            self.node_group.setEnabled(False)
            self.member_group.setEnabled(self.gl_widget.selected_member is not None)
        else:
            self.node_group.setEnabled(False)
            self.member_group.setEnabled(False)

        self.update_info_text()

    def on_node_selected(self, node_id):
        """Handle node selection"""
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

    def on_member_selected(self, member_id):
        """Handle member selection"""
        debug_print(f"Member {member_id} selected")

        if self.current_mode == "delete_member":
            self.delete_member(member_id)
            return

        # Find member data
        member = next((m for m in self.members if m['id'] == member_id), None)
        if not member:
            return

        # Update member properties panel
        self.member_id_label.setText(str(member_id))
        self.member_nodes_label.setText(f"{member['i_node']} - {member['j_node']}")

        # Block signals
        self.member_section_combo.blockSignals(True)
        self.member_material_combo.blockSignals(True)

        self.member_section_combo.setCurrentIndex(member['section'] - 1)
        self.member_material_combo.setCurrentIndex(member['material'] - 1)

        self.member_section_combo.blockSignals(False)
        self.member_material_combo.blockSignals(False)

        # Enable/disable panels based on mode
        self.node_group.setEnabled(False)
        if self.current_mode == "edit_member":
            self.member_group.setEnabled(True)

        self.update_info_text()

    def update_node_position(self):
        """Update selected node position from spinboxes"""
        if not self.gl_widget.selected_node:
            return

        node_id = self.gl_widget.selected_node
        x = self.node_x_spin.value()
        y = self.node_y_spin.value()
        z = self.node_z_spin.value()

        # Update internal data
        self.node_dict[node_id] = (x, y, z)

        # Update node in list
        for node in self.nodes:
            if node['id'] == node_id:
                node['x'] = f"{x} ft"
                node['y'] = f"{y} ft"
                node['z'] = f"{z} ft"
                break

        # Update GL widget
        self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)

    def update_member_properties(self):
        """Update selected member properties"""
        if not self.gl_widget.selected_member:
            return

        member_id = self.gl_widget.selected_member

        # Find member
        for member in self.members:
            if member['id'] == member_id:
                member['section'] = self.member_section_combo.currentIndex() + 1
                member['material'] = self.member_material_combo.currentIndex() + 1
                break

        # Update GL widget
        self.gl_widget.update()

    def start_change_member_nodes(self):
        """Start process to change member's connected nodes"""
        if not self.gl_widget.selected_member:
            return

        # Switch to special mode for changing member nodes
        self.current_mode = "change_member_nodes"
        self.member_creation_nodes = []

        QMessageBox.information(self, "Change Nodes",
                              "Select two nodes to reconnect the member")
        self.update_info_text()

    def update_member_nodes(self, member_id, new_i_node, new_j_node):
        """Update a member's connected nodes"""
        member = next((m for m in self.members if m['id'] == member_id), None)
        if not member:
            return

        old_i_node = member['i_node']
        old_j_node = member['j_node']

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

        # Update GL widget
        self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)

    def create_member(self, node1, node2):
        """Create a new member between two nodes"""
        section = self.section_combo.currentIndex() + 1
        material = self.material_combo.currentIndex() + 1

        # Get next member ID
        next_id = max([m['id'] for m in self.members], default=0) + 1

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

        # Update visualization
        self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)
        self.populate_structure_tree()

    def delete_node(self, node_id):
        """Delete a node and its connected members"""
        debug_print(f"Deleting node {node_id}")

        # Get connected members
        connected_members = self.node_connections.get(node_id, [])[:]

        # Delete connected members
        if connected_members:
            for member_id in connected_members:
                self.delete_member(member_id, update_viz=False)

        # Delete node
        self.nodes = [n for n in self.nodes if n['id'] != node_id]
        if node_id in self.node_dict:
            del self.node_dict[node_id]
        if node_id in self.node_connections:
            del self.node_connections[node_id]

        # Update GL widget
        self.gl_widget.selected_node = None
        self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)
        self.populate_structure_tree()
        self.update_info_text()

    def delete_member(self, member_id, update_viz=True):
        """Delete a member"""
        debug_print(f"Deleting member {member_id}")

        # Find member
        member = next((m for m in self.members if m['id'] == member_id), None)
        if not member:
            return

        # Remove from connections
        i_node = member['i_node']
        j_node = member['j_node']

        if i_node in self.node_connections and member_id in self.node_connections[i_node]:
            self.node_connections[i_node].remove(member_id)

        if j_node in self.node_connections and member_id in self.node_connections[j_node]:
            self.node_connections[j_node].remove(member_id)

        # Remove from list
        self.members = [m for m in self.members if m['id'] != member_id]

        # Update visualization if requested
        if update_viz:
            self.gl_widget.selected_member = None
            self.gl_widget.setStructureData(self.node_dict, self.members, self.supports)
            self.populate_structure_tree()
            self.update_info_text()

    def update_info_text(self):
        """Update information text"""
        info = f"Mode: {self.current_mode.replace('_', ' ').title()}\n\n"

        if self.current_mode == "add_member":
            info += f"Select nodes to create a member.\n"
            info += f"Selected nodes: {len(self.member_creation_nodes)}/2\n\n"
        elif self.current_mode == "change_member_nodes":
            info += f"Select nodes to reconnect member {self.gl_widget.selected_member}.\n"
            info += f"Selected nodes: {len(self.member_creation_nodes)}/2\n\n"

        if self.gl_widget.selected_node:
            node_id = self.gl_widget.selected_node
            x, y, z = self.node_dict[node_id]
            info += f"Selected Node: {node_id}\n"
            info += f"Position: ({x:.2f}, {y:.2f}, {z:.2f})\n"
            info += f"Connected Members: {len(self.node_connections[node_id])}\n"

        elif self.gl_widget.selected_member:
            member = next((m for m in self.members if m['id'] == self.gl_widget.selected_member), None)
            if member:
                info += f"Selected Member: {member['id']}\n"
                info += f"Nodes: {member['i_node']} - {member['j_node']}\n"
                info += f"Section: {member['section']}, Material: {member['material']}\n"

                # Calculate length
                i_pos = self.node_dict[member['i_node']]
                j_pos = self.node_dict[member['j_node']]
                length = sp.sqrt(sum((j_pos[i] - i_pos[i])**2 for i in range(3)))
                info += f"Length: {length:.2f} ft\n"

        self.info_text.setText(info)

    def reset_view(self):
        """Reset camera view in GL widget"""
        debug_print("Resetting view")
        self.gl_widget.resetView()

    def save_structure(self):
        """Save structure to JSON file"""
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

def main():
    """Main entry point"""
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
                print("No JSON files found")
                sys.exit(1)

    # Create and show main window
    try:
        editor = StairwayInteractiveEditor(json_file)
        editor.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
