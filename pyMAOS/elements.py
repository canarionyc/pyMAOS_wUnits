# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod

import pyMAOS.loading as loadtypes


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
                f"Material({self.material}), Section({self.section})")
    
    def __repr__(self):
        """Return developer representation of the element"""
        return self.__str__()

    @property
    def length(self):
        """Calculate member length from the i and j nodes"""
        return self.inode.distance(self.jnode)
        
    def T(self) -> np.matrix:
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
        ], dtype=float)
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
        
    def kglobal(self):
        """Calculate the global stiffness matrix for the element"""
        k = self.k()
        T = self.T(); print("T:\n", T)
        global_stiffness_matrix = np.matmul(np.matmul(np.transpose(T), k), T); print("kglobal:", global_stiffness_matrix)
        return global_stiffness_matrix

    def Dglobal(self, load_combination):
        """Get global nodal displacement vector for the element"""
        D = np.zeros(6)
        iD = self.inode.displacements[load_combination.name]
        jD = self.jnode.displacements[load_combination.name]

        # Populate Displacement Vector
        displacement_vector_global = np.array([*iD, *jD])
        
        # Import and use the display function if it's available
        try:
            from pyMAOS.display_utils import display_member_displacement_in_units
            if hasattr(self, 'structure') and hasattr(self.structure, 'units'):
                force_unit = self.structure.units.get('force', 'N')
                length_unit = self.structure.units.get('length', 'm')
            else:
                # Default units if structure reference not available
                force_unit = 'N'
                length_unit = 'm'
            
            display_member_displacement_in_units(
                displacement_vector_global,
                force_unit=force_unit,
                length_unit=length_unit,
                is_local=False,
                element_uid=self.uid,
                load_combo_name=load_combination.name
            )
        except ImportError:
            print(f"Global displacements for element {self.uid} under load combination '{load_combination.name}':\n{displacement_vector_global}")

        return displacement_vector_global

    def Dlocal(self, load_combination):
        """Calculate local displacement vector"""
        Dglobal = self.Dglobal(load_combination)
        displacement_vector_local = np.matmul(self.T(), Dglobal)
        
        # Import and use the display function if it's available
        try:
            from pyMAOS.display_utils import display_member_displacement_in_units
            if hasattr(self, 'structure') and hasattr(self.structure, 'units'):
                force_unit = self.structure.units.get('force', 'N')
                length_unit = self.structure.units.get('length', 'm')
            else:
                # Default units if structure reference not available
                force_unit = 'N'
                length_unit = 'm'
            
            display_member_displacement_in_units(
                displacement_vector_local,
                force_unit=force_unit,
                length_unit=length_unit,
                is_local=True,
                element_uid=self.uid,
                load_combo_name=load_combination.name
            )
        except ImportError:
            print(f"Local displacements for element {self.uid} under load combination '{load_combination.name}':\n{displacement_vector_local}")

        return displacement_vector_local

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




