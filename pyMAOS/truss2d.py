# -*- coding: utf-8 -*-
import numpy as np

import pyMAOS.loading as loadtypes
from pyMAOS.elements import Element

class R2Truss(Element):
    def __init__(self, uid, inode, jnode, material, section):
        super().__init__(uid, inode, jnode, material, section)
        self.type = "TRUSS"
        self.isoff = {}
        self._TensionOnly = False
        self._CompressionOnly = False

        # Validate truss after initialization
        self.validate_truss()

    def set_tension_only(self):
        self._TensionOnly = True
        self.clear_compression_only()

    def set_compression_only(self):
        self._CompressionOnly = True
        self.clear_tension_only()

    def clear_tension_only(self):
        self._TensionOnly = False

    def clear_compression_only(self):
        self._CompressionOnly = False

    def k(self):
        """Calculate the local stiffness matrix for the truss element
        
        Creates a 6x6 stiffness matrix with only axial terms active.
        All bending/shear terms are zero since truss elements can only
        transfer axial forces.
        
        Returns
        -------
        numpy.matrix
            6x6 local stiffness matrix for the truss element
        """
        E = self.material.E
        A = self.section.Area
        L = self.length
        
        # Initialize matrix with zeros
        k = np.zeros((6, 6))
        
        # Only axial terms are non-zero in truss elements
        axial_stiffness = (A * E) / L
        
        # Diagonal terms (positive)
        k[0, 0] = k[3, 3] = axial_stiffness
        
        # Off-diagonal terms (negative)
        k[0, 3] = k[3, 0] = -axial_stiffness
        
        return np.matrix(k)

    def Flocal(self, load_combination):
        """Calculate element end forces in the local coordinate system
        
        Computes the forces at each end of the element based on nodal 
        displacements and stores them in the end_forces_local dictionary.
        
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to calculate forces
        """
        Dlocal = self.set_displacement_local(load_combination)

        FL = np.matmul(self.k(), Dlocal.T)
        print(f"Element {self.uid} end forces in local coordinates: {FL}")
        self.end_forces_local[load_combination.name] = FL

    def Fglobal(self, load_combination):
        Dglobal = self.set_displacement_global(load_combination)

        # global stiffness matrix
        KG = self.kglobal()

        FG = np.matmul(KG, Dglobal)

        self.end_forces_global[load_combination.name] = FG
        print(f"Element {self.uid} end forces in global coordinates: {FG}")
        self.Flocal(load_combination)

        return FG

    def Alocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecewise functions to build the total function from the loading
        ax = loadtypes.PiecewisePolynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)

        ax = ax.combine(fxi.Ax, 1, 1)
        ax = ax.combine(fxj.Ax, 1, 1)

        axlocal_span = np.zeros((len(self.calcstations), 2))

        for i, x in enumerate(self.calcstations):
            a = ax.evaluate(x)

            axlocal_span[i, 0] = x
            axlocal_span[i, 1] = a * scale

        return axlocal_span

    def Aglobal_plot(self, load_combination, scale):
        axlocal_plot = self.Alocal_plot(load_combination, scale=scale)

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        axglobal_plot = np.matmul(axlocal_plot, R)

        return axglobal_plot

    def Dlocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        Dlocal = self.set_displacement_local(load_combination)

        # Parametric Functions defining a linear relationship for deflection
        # in each axis based on the Ux and Uy nodal displacements
        Dx = lambda x: Dlocal[0, 0] + (x / self.length) * (Dlocal[0, 3] - Dlocal[0, 0])
        Dy = lambda x: Dlocal[0, 1] + (x / self.length) * (Dlocal[0, 4] - Dlocal[0, 1])

        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecewise functions to build the total function from the loading
        dx = loadtypes.PiecewisePolynomial()
        dy = loadtypes.PiecewisePolynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)

        dx = dx.combine(fxi.Dx, 1, 1)
        dx = dx.combine(fxj.Dx, 1, 1)

        dlocal_span = np.zeros((len(self.calcstations), 2))

        for i, x in enumerate(self.calcstations):
            dxl = dx.evaluate(x) + Dx(0)
            dyl = dy.evaluate(x) + Dy(x)

            dlocal_span[i, 0] = x + (dxl * scale)
            dlocal_span[i, 1] = dyl * scale

        return dlocal_span

    def Dglobal_plot(self, load_combination, scale=1):
        dlocal_plot = self.Dlocal_plot(load_combination, scale=scale)

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        dglobal_plot = np.matmul(dlocal_plot, R)

        return dglobal_plot
    
    def validate_truss(self):
        """
        Validates that the truss element is consistent with the node restraints.
        Raises an error if there is a conflict.
        """
        # Check i-node (start node)
        if self.inode.restraints[2] == 0:
            raise ValueError(
                f"Conflict: Truss element (UID: {self.uid}) connected to i-node (UID: {self.inode.uid}) "
                f"has a no rotational restraint (Rz = 0), which is not allowed for trusses."
            )

        # Check j-node (end node)
        if self.jnode.restraints[2] == 0:
            raise ValueError(
                f"Conflict: Truss element (UID: {self.uid}) connected to j-node (UID: {self.jnode.uid}) "
                f"has a no rotational restraint (Rz = 0), which is not allowed for trusses."
            )

    # New method to disallow distributed loads for truss elements
    def add_distributed_load(self, *args, **kwargs):
        raise ValueError(
            "Distributed loads are not allowed for truss elements. " +
            "Only FRAME elements support distributed loads."
        )
