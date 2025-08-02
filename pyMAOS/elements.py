import numpy as np
from abc import ABC, abstractmethod

from numpy import ndarray

import pyMAOS.loading as loadtypes
import quantity_utils
from pyMAOS.units_mod import unit_manager
from quantity_utils import QuantityArray, quantity_array_to_float64


# In structural analysis, hinges in frame elements (like beams and columns) serve a distinct purpose from node restraints. Here's what they do:
# Node restraints (rx, ry, rz) control whether a node can move or rotate in the global coordinate system. These apply to the node itself.
# Member hinges, however, create a release condition at the connection between a member and a node, allowing the member to rotate independently of the node's rotation. This is useful for modeling:
# 1.	Pin connections where moment cannot be transferred between members
# 2.	Partially fixed connections with limited moment transfer
# 3.	Construction details like simple beam-to-column connections
# 4.	Plastic hinge formation in advanced analysis
# The question mentions "node hinges" but the code actually deals with member end hinges (as seen in the debug output where it prints element.hinges). These allow accurate modeling of connection behavior, which is essential for proper force distribution in the structure.
# The comment about redundancy with restraints suggests there might be some confusion about the distinct roles of these features - restraints control global node behavior while hinges control member-to-node connectivity.

class Element(ABC):
    """Base class for structural elements"""
    
    def __init__(self, uid, inode, jnode, material, section):
        self.uid = uid
        self.inode = inode
        self.jnode = jnode
        self.material = material
        self.section = section
        self.end_forces_local = {}
        self.end_forces_global = {}
        self._stations = False
        self.type = "GENERIC"  # Default type, will be overridden by derived classes

    def __str__(self):
        """Return string representation of the element"""
        return (f"{self.type} Element {self.uid}: "
                f"Nodes({self.inode.uid}->{self.jnode.uid}), "
                f"{self.material}, {self.section}")
    
    def __repr__(self):
        """Return developer representation of the element"""
        return (f"Element(uid={self.uid}, type={self.type}, inode={self.inode.uid}, jnode={self.jnode.uid}, material={self.material.uid}, section={self.section.uid})")

    @property
    def length(self):
        """Calculate member length from the i and j nodes"""
        return self.inode.distance(self.jnode)
        
    def set_rotation_matrix(self) -> np.matrix:
        """Create transformation matrix from local to global coordinates
    
        Returns
        -------
        np.matrix
            6x6 transformation matrix of float type for converting between 
            local and global coordinate systems
        """
        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        T = np.matrix([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)
        self.rotation_matrix = T
        return T

    @abstractmethod
    def k(self) -> np.matrix:
        """Calculate the local stiffness matrix for the element.
    
        This method must be implemented by all derived classes.
    
        Returns
        -------
        numpy.matrix
            The local stiffness matrix
        """
        pass
    # def k_with_units(self):
    #     """Calculate the local stiffness matrix with units"""
    #     k = self.k()
    #     # This is a placeholder implementation. Actual unit handling would depend on the specific units used in the analysis.
    #     # For example, you might convert the matrix to a specific unit system here.
    #     return k
    def kglobal(self):
        """Calculate the global stiffness matrix for the element

        Transforms the local stiffness matrix to global coordinates using
        the transformation matrix T: kglobal = T^T * klocal * T

        Returns
        -------
        ndarray
            Element stiffness matrix in global coordinates
        """
        # Get transformation matrix
        T = self.set_rotation_matrix()

        local_k = self.k()
        localk_float64 = quantity_array_to_float64(local_k)
        # Perform the transformation while preserving units
        from numpy import linalg
        # Transform local stiffness to global coordinates efficiently
        globalk_float64 = linalg.multi_dot([T.T, localk_float64, T])

        print(f"DEBUG: Element {self.uid} global stiffness matrix created with shape {globalk_float64.shape}")

        print(f"Element {self.uid} global stiffness matrix created with shape {globalk_float64.shape}")

        return globalk_float64

    def display_stiffness_matrix_in_units(self):
        """Display the stiffness matrix with appropriate units notation."""
        # This is a placeholder implementation. Actual unit handling would depend on the specific units used in the analysis.
        k_with_units = self.k_with_units()
        print(f"Stiffness Matrix for Element {self.uid}:\n")
        print(k_with_units)

    def set_displacement_global(self, load_case):
        """Get global nodal displacement vector for the element"""

        iD = self.inode.displacements[load_case]; print(f"iD: {iD}")
        jD = self.jnode.displacements[load_case]; print(f"jD: {jD}")

        # Convert each QuantityArray to numpy arrays of Quantities
        # from pyMAOS.quantity_utils import quantity_array_to_numpy
        # iD_array = quantity_array_to_numpy(iD)
        # jD_array = quantity_array_to_numpy(jD)

        # Now you can create a combined displacement vector
        self.displacement_global = np.concatenate([iD, jD])
        return self.displacement_global

    def set_displacement_local(self, load_case):
        """Calculate local displacement vector"""
        elem_global_displacement = self.set_displacement_global(load_case)
        from pyMAOS import quantity_utils
        from numpy import linalg
        from pyMAOS.quantity_utils import quantity_array_to_float64
        self.displacement_local = np.dot(self.set_rotation_matrix(), convert_array_to_float64(elem_global_displacement))
      
        return self.displacement_local

    def stations(self, num_stations=10):
        """Define calculation points along the element"""
        # This is a basic implementation - derived classes may override
        eta = [0 + i * (1 / num_stations) for i in range(num_stations + 1)]
        stations = [self.length * i for i in eta]
        
        # Make sure the first and last stations do not exceed the beam
        if stations[0] < 0:
            stations[0] = 0
        if stations[-1] > self.length:
            stations[-1] = self.length

        # Remove duplicate locations
        self.calcstations = sorted(set(stations))
        self._stations = True

    def set_structure(self, structure):
        """Attach reference to parent structure for unit access"""
        self.structure = structure

