# -*- coding: utf-8 -*-
import os
import numpy as np

from pyMAOS.display_utils import display_node_load_vector_in_units, display_stiffness_matrix_in_units
import pyMAOS.loading as loadtypes
from pyMAOS.elements import Element

class R2Frame(Element):
    def __init__(self, uid, inode, jnode, material, section):
        super().__init__(uid, inode, jnode, material, section)
        self.type = "FRAME"
        self.hinges = [0, 0]
        self.loads = []
        self.fixed_end_forces = {}

        # Internal Functions
        # Dictionary key for each combination
        self.Wx = {}
        self.Wy = {}
        self.A = {}
        self.Vy = {}
        self.Mz = {}
        self.Sz = {}
        self.Dx = {}
        self.Dy = {}

        # Flags
        self._loaded = False

        # Validate hinges after initialization
        self.validate_hinges()

    def __str__(self):
        """Return string representation of the frame element including hinge status"""
        base_str = super().__str__()
        hinge_info = ""
    
        if self.hinges[0] == 1 and self.hinges[1] == 1:
            hinge_info = ", Hinges: Both ends"
        elif self.hinges[0] == 1:
            hinge_info = ", Hinge: Start node"
        elif self.hinges[1] == 1:
            hinge_info = ", Hinge: End node"
        
        return base_str + hinge_info

    def hinge_i(self):
        """Apply a moment release (hinge) at the start node of the frame element
        
        Creates a rotational release at the i-node connection, which prevents 
        moment transfer between the element and the node. The hinge status is 
        validated to ensure it doesn't conflict with node restraints.
        
        Notes
        -----
        Hinges create pin connections that allow free rotation at element ends.
        This affects the stiffness matrix and the distribution of forces in the element.
        """
        self.hinges[0] = 1
        self.validate_hinges()

    def hinge_j(self):
        """Apply a moment release (hinge) at the end node of the frame element
        
        Creates a rotational release at the j-node connection, which prevents
        moment transfer between the element and the node. The hinge status is
        validated to ensure it doesn't conflict with node restraints.
        
        Notes
        -----
        Hinges create pin connections that allow free rotation at element ends.
        This affects the stiffness matrix and the distribution of forces in the element.
        """
        self.hinges[1] = 1
        self.validate_hinges()

    def fix_i(self):
        """Remove hinge at the start node of the frame element
        
        Restores full moment continuity between the element and its i-node.
        This allows moment transfer at the connection.
        """
        self.hinges[0] = 0

    def fix_j(self):
        """Remove hinge at the end node of the frame element
        
        Restores full moment continuity between the element and its j-node.
        This allows moment transfer at the connection.
        """
        self.hinges[1] = 0

    def add_point_load(self, p, a, case="D", direction="y", location_percent=False):
        """Add a concentrated point load to the frame element
        
        Parameters
        ----------
        p : float
            Magnitude of the point load
        a : float
            Position along the element where the load is applied.
            If location_percent is True, this is a percentage (0-100)
        case : str, optional
            Load case identifier, default is "D" (dead load)
        direction : str, optional
            Direction of the load: "x", "y", "X", or "Y", default is "y"
            Lowercase indicates local axes, uppercase indicates global axes
            "xx" for axial load in local x direction
        location_percent : bool, optional
            If True, 'a' is interpreted as percentage of element length
            
        Returns
        -------
        None
        
        Notes
        -----
        For global loads (X, Y), the load is transformed into local components
        and stored as two separate loads (axial and transverse)
        """
        if location_percent:
            a = (a / 100) * self.length
        if direction == "Y" or direction == "X":
            # Load is applied in the global axis

            c = (self.jnode.x - self.inode.x) / self.length
            s = (self.jnode.y - self.inode.y) / self.length

            if direction == "Y":
                pyy = c * p
                pxx = s * p
            else:
                pyy = -1 * s * p
                pxx = c * p
            self.loads.append(loadtypes.R2_Axial_Load(pxx, a, self, loadcase=case))
            self.loads.append(loadtypes.R2_Point_Load(pyy, a, self, loadcase=case))
        else:
            # Load is applied in the local member axis

            if direction == "xx":
                self.loads.append(loadtypes.R2_Axial_Load(p, a, self, loadcase=case))
            else:
                self.loads.append(loadtypes.R2_Point_Load(p, a, self, loadcase=case))

        self._stations = False
        self._loaded = True

    def add_distributed_load(self, wi, wj, a, b, case="D", direction="y", location_percent=False, projected=False):
        """Add a distributed load to the frame element
        
        Parameters
        ----------
        wi : float
            Starting intensity of the distributed load
        wj : float
            Ending intensity of the distributed load
        a : float
            Starting position of the distributed load
        b : float
            Ending position of the distributed load
        case : str, optional
            Load case identifier, default is "D" (dead load)
        direction : str, optional
            Load direction: "x", "y", "X", or "Y", default is "y"
            Lowercase indicates local axes, uppercase indicates global axes
        location_percent : bool, optional
            If True, a and b are interpreted as percentages of element length
        projected : bool, optional
            If True, load intensity is based on projected length
            
        Notes
        -----
        For global loads (X, Y), the load is transformed into local components
        and stored as two separate loads
        """
        if location_percent:
            a = (a / 100) * self.length
            b = (b / 100) * self.length
        if direction == "Y" or direction == "X":
            # Load is applied in the global axis

            c = (self.jnode.x - self.inode.x) / self.length
            s = (self.jnode.y - self.inode.y) / self.length

            if direction == "Y":
                if projected:
                    wi = c * wi
                    wj = c * wj

                wyyi = c * wi
                wyyj = c * wj
                wxxi = s * wi
                wxxj = s * wj
            else:
                if projected:
                    wi = s * wi
                    wj = s * wj

                wyyi = -1 * s * wi
                wyyj = -1 * s * wj
                wxxi = c * wi
                wxxj = c * wj
            if abs(wxxi) > 1e-10 or abs(wxxj) > 1e-10:  # Only add if at least one component is non-zero
                self.loads.append(
                    loadtypes.R2_Axial_Linear_Load(wxxi, wxxj, a, b, self, loadcase=case)
                )
            if abs(wyyi) > 1e-10 or abs(wyyj) > 1e-10:  # Also check transverse load
                self.loads.append(
                    loadtypes.R2_Linear_Load(wyyi, wyyj, a, b, self, loadcase=case)
                )
        else:
            # Load is applied in the local member axis

            if direction == "xx":
                self.loads.append(
                    loadtypes.R2_Axial_Linear_Load(wi, wj, a, b, self, loadcase=case)
                )
            else:
                self.loads.append(
                    loadtypes.R2_Linear_Load(wi, wj, a, b, self, loadcase=case)
                )

        self._stations = False
        self._loaded = True

    def add_moment_load(self, m, a, case="D", location_percent=False):
        """Add a concentrated moment to the frame element
        
        Parameters
        ----------
        m : float
            Magnitude of the moment (positive according to right-hand rule)
        a : float
            Position along the element where the moment is applied.
            If location_percent is True, this is a percentage (0-100)
        case : str
            Load case identifier (e.g., "D" for dead load, "L" for live load)
        location_percent : bool, optional
            If True, 'a' is interpreted as percentage of element length, default is False
            
        Returns
        -------
        None
        
        Notes
        -----
        Positive moments follow the right-hand rule convention (counterclockwise).
        The moment is applied in the local coordinate system of the element.
        """
        if location_percent:
            a = (a / 100) * self.length
        self.loads.append(loadtypes.R2_Point_Moment(m, a, self, loadcase=case))

        self._stations = False
        self._loaded = True

    def FEF(self, load_combination):
        """Calculate Fixed End Forces for the element under the given load combination
        
        Fixed End Forces (FEF) represent the equivalent nodal forces due to member loads.
        These are the forces needed at the element ends to maintain equilibrium when 
        the element is subjected to distributed loads, point loads, or moments applied
        along its length.
        
        The method processes all loads assigned to the element, applies the appropriate
        load factors from the load combination, and combines them into a single force vector.
        
        If element has hinges, the fixed end forces are modified to account for the releases.
        
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination object containing load case factors
            
        Returns
        -------
        numpy.ndarray
            6-element vector of fixed end forces in local coordinates:
            [Fi_x, Fi_y, Mi_z, Fj_x, Fj_y, Mj_z]
            where i = start node, j = end node, and x,y,z are local coordinates
        """
        import sys
        
        # Initialize fixed end forces vector
        fef = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Process each load applied to the element
        print(f"Processing {len(self.loads)} loads on element {self.uid}:", file=sys.stderr)
        for load_idx, load in enumerate(self.loads):
            load_case = load.loadcase
            load_factor = load_combination.factors.get(load_case, 0)
            
            # Skip loads that don't contribute to this combination
            if load_factor == 0:
                print(f"  Load {load_idx}: {load.kind} (case '{load_case}') - skipped (factor=0)", file=sys.stderr)
                continue
                
            # Calculate FEF contribution from this load
            load_fef = load.FEF()
            factored_fef = np.array([load_factor * i for i in load_fef])

            print(f"Element {self.uid} load {load_idx}: {load.kind} (case '{load_case}', factor={load_factor})", file=sys.stderr)
            print(f"    Raw FEF: {load_fef}", file=sys.stderr)
            print(f"    Factored: {factored_fef}", file=sys.stderr)

            # Add to total FEF
            fef = fef + factored_fef
        
        # Handle hinge conditions - these modify the fixed end forces for partial releases
        if self.hinges == [1, 0]:  # Hinge at start node
            Mi = fef[2]
            L = self.length
            
            print(f"  Applying hinge at start node - redistributing moment Mi={Mi}", file=sys.stderr)
            fef[1] = fef[1] - ((3 / (2 * L)) * Mi)
            fef[2] = 0  # Zero moment at hinge
            fef[4] = fef[4] + ((3 / (2 * L)) * Mi)
            fef[5] = fef[5] - (Mi / 2)
            
        elif self.hinges == [0, 1]:  # Hinge at end node
            Mj = fef[5]
            L = self.length
            
            print(f"  Applying hinge at end node - redistributing moment Mj={Mj}", file=sys.stderr)
            fef[1] = fef[1] - ((3 / (2 * L)) * Mj)
            fef[2] = fef[2] - (Mj / 2)
            fef[4] = fef[4] + ((3 / (2 * L)) * Mj)
            fef[5] = 0  # Zero moment at hinge
            
        elif self.hinges == [1, 1]:  # Hinges at both nodes
            Mi = fef[2]
            Mj = fef[5]
            L = self.length
            
            print(f"  Applying hinges at both nodes - redistributing moments Mi={Mi}, Mj={Mj}", file=sys.stderr)
            fef[1] = fef[1] - ((Mj + Mi) / L)
            fef[2] = 0  # Zero moment at hinge
            fef[4] = fef[4] + ((Mj + Mi) / L)
            fef[5] = 0  # Zero moment at hinge

        print(f"  Final FEF for element {self.uid}: {fef}", file=sys.stderr)
        display_node_load_vector_in_units(fef[0:3], 
                                          force_unit=self.structure.units['force'], 
                                          length_unit=self.structure.units['length'], 
                                          load_combo_name=None)
        display_node_load_vector_in_units(fef[3:6], 
                                          force_unit=self.structure.units['force'], 
                                          length_unit=self.structure.units['length'], 
                                          load_combo_name=None)
        return fef

    def FEFglobal(self, load_combination):
        """Calculate fixed end forces in the global coordinate system
    
        Transforms the local fixed end forces to the global coordinate system
        using the element's transformation matrix.
    
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination object containing load case factors
        
        Returns
        -------
        numpy.ndarray
            6-element vector of fixed end forces in global coordinates:
            [Fi_x, Fi_y, Mi_z, Fj_x, Fj_y, Mj_z]
            where i = start node, j = end node, and x,y,z are global coordinates
    
        Notes
        -----
        The global fixed end forces are used when assembling the global
        load vector during structural analysis.
        """ 

        fef = np.transpose(self.FEF(load_combination))
        T = self.T()
        ret_val= np.matmul(np.transpose(T), fef)
        print("Global fixed end forces (ret_val):", ret_val)
        return ret_val.A.flatten()

    def k(self, **kwargs):
        """Calculate the local stiffness matrix for the frame element
        
        Creates a 6x6 stiffness matrix with appropriate modifications
        for hinges if present. The stiffness matrix includes terms for
        axial, shear, and bending behavior.
        
              [ EA/L       0         0      -EA/L       0         0     ]
              [   0     12EI/L³   6EI/L²      0      -12EI/L³   6EI/L² ]
        [k] = [   0     6EI/L²    4EI/L       0      -6EI/L²    2EI/L  ]
              [-EA/L       0         0       EA/L       0         0     ]
              [   0    -12EI/L³  -6EI/L²      0       12EI/L³  -6EI/L² ]
              [   0     6EI/L²    2EI/L       0      -6EI/L²    4EI/L  ]

        Different matrices are used depending on hinge configuration:
        - No hinges: Standard beam element
        - Hinge at i-end: Modified for released moment at i
        - Hinge at j-end: Modified for released moment at j
        - Hinges at both ends: Truss-like behavior with no moment transfer
        
        Returns
        -------
        numpy.matrix
            6x6 local stiffness matrix for the frame element
        """
        E = self.material.E
        Ixx = self.section.Ixx
        A = self.section.Area
        L = self.length
        
        if self.uid==2:
            print(f"Calculating local stiffness matrix for element {self.uid} with length {L}, E={E}, Ixx={Ixx}, A={A}, hinges={self.hinges}")

        # Initialize matrix with zeros
        k = np.zeros((6, 6))
        
        # Common terms to improve readability
        AE_L = A * E / L
        EI = E * Ixx
        EI_L = EI / L
        EI_L2 = EI / (L * L)
        EI_L3 = EI / (L * L * L)
        
        # Axial terms (common to all hinge configurations)
        k[0, 0] = k[3, 3] = AE_L
        k[0, 3] = k[3, 0] = -AE_L
        
        # Apply appropriate bending terms based on hinge configuration
        if self.hinges == [1, 1]:  # Both ends hinged - only axial stiffness
            pass  # No additional terms needed
            
        elif self.hinges == [1, 0]:  # Hinge at i-end
            k[1, 1] = k[4, 4] = 3 * EI_L3
            k[1, 4] = k[4, 1] = -3 * EI_L3
            k[1, 5] = k[5, 1] = 3 * EI_L2
            k[4, 5] = k[5, 4] = -3 * EI_L2
            k[5, 5] = 3 * EI_L
            
        elif self.hinges == [0, 1]:  # Hinge at j-end
            k[1, 1] = k[4, 4] = 3 * EI_L3
            k[1, 2] = k[2, 1] = 3 * EI_L2
            k[1, 4] = k[4, 1] = -3 * EI_L3
            k[2, 4] = k[4, 2] = -3 * EI_L2
            k[2, 2] = 3 * EI_L
            
        else:  # No hinges - standard beam element
            k[1, 1] = k[4, 4] = 12 * EI_L3
            k[1, 2] = k[2, 1] = 6 * EI_L2
            k[1, 4] = k[4, 1] = -12 * EI_L3
            k[1, 5] = k[5, 1] = 6 * EI_L2
            k[2, 2] = 4 * EI_L
            k[2, 4] = k[4, 2] = -6 * EI_L2
            k[2, 5] = k[5, 2] = 2 * EI_L
            k[4, 5] = k[5, 4] = -6 * EI_L2
            k[5, 5] = 4 * EI_L
     
        local_stiffness_matrix = np.matrix(k)
        # print(f"Local stiffness matrix for element {self.uid}:\n{local_stiffness_matrix}")

        # First try to get units from element's structure
        units_dict = None
        
        # Option 1: Check if element has direct reference to structure with units
        if hasattr(self, 'structure') and hasattr(self.structure, 'units'):
            units_dict = self.structure.units
        
        # Option 2: Try to get units from units module
        if units_dict is None:
            try:
                from pyMAOS.units import DISPLAY_UNITS
                units_dict = DISPLAY_UNITS
            except ImportError:
                # Default to SI units if import fails
                units_dict = {'force': 'N', 'length': 'm'}

        from pyMAOS.display_utils import display_stiffness_matrix_in_units; print(f"Local stiffness matrix in units for element {self.uid}:\n")
        k_converted=display_stiffness_matrix_in_units(k, force_unit=units_dict.get('force', 'N'), length_unit=units_dict.get('length', 'm'), return_matrix=True)
        k_converted_csv=os.path.join(os.getcwd(), f"local_stiffness_matrix_{self.uid}.txt")
        np.savetxt(k_converted_csv, k_converted, fmt='%.6f')
        return local_stiffness_matrix

    def Flocal(self, load_combination):
        """Calculate element end forces in the local coordinate system
        
        Computes the end forces by combining:
        1. Forces due to nodal displacements (k*d)
        2. Fixed end forces due to applied loads
        
        The result is stored in the end_forces_local dictionary.
        
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to calculate forces
            
        Notes
        -----
        This method calculates internal member forces in the element's local 
        coordinate system, which is oriented along the member's axis.
        """
        Dlocal = self.Dlocal(load_combination)
        Qf = np.reshape(self.FEF(load_combination), (-1, 1))

        FL = np.matmul(self.k(), Dlocal.T)

        self.end_forces_local[load_combination.name] = FL + Qf

    def Fglobal(self, load_combination):
        """Calculate element end forces in the global coordinate system
        
        Computes the global end forces by combining:
        1. Forces due to nodal displacements (KG*D)
        2. Fixed end forces transformed to global coordinates
        
        The result is stored in the end_forces_global dictionary.
        This method also updates the local end forces by calling Flocal().
        
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to calculate forces
            
        Returns
        -------
        numpy.ndarray
            6-element vector of end forces in global coordinates:
            [Fi_x, Fi_y, Mi_z, Fj_x, Fj_y, Mj_z]
            where i = start node, j = end node
        """
        Dglobal = self.Dglobal(load_combination)
        
        Qfg = self.FEFglobal(load_combination)
        print(f"Global fixed end forces for element {self.uid} under load combination '{load_combination.name}':\n{Qfg}")
        print(f"Calculating global displacements for element {self.uid} under load combination '{load_combination.name}'")
        
        # global stiffness matrix
        KG = self.kglobal()
        print(f"Global stiffness matrix for element {self.uid}:\n{KG}")
        display_stiffness_matrix_in_units(KG, force_unit='N', length_unit='m', return_matrix=False)
        FG = np.matmul(KG, Dglobal).A.flatten()
        print(f"Global end forces for element {self.uid} under load combination '{load_combination.name}':\n{FG}")

        # Store the global end forces
        if not hasattr(self, 'end_forces_global'):
            self.end_forces_global = {}

        ret_val= FG + Qfg
        self.end_forces_global[load_combination.name] = ret_val
        print(f"End forces in local coordinates for element {self.uid} under load combination '{load_combination.name}':\n{self.end_forces_local.get(load_combination.name, 'Not calculated')}")
        self.Flocal(load_combination)

        return ret_val

    def stations(self, num_stations=10):
        """
        Define evenly distributed points along the member to compute internal
        actions. Additional points are generated for load application points.

        This also generates a reduced set of points for use in the max/min
        internal action functions.

        :param num_stations: _description_, defaults to 10
        :type num_stations: int, optional
        """

        # parametric list of stations between 0 and 1'
        eta = [0 + i * (1 / num_stations) for i in range(num_stations + 1)]

        stations = [self.length * i for i in eta]
        max_stations = [0, self.length]

        if self._loaded:
            extra_stations = []

            for load in self.loads:
                if (
                    load.kind == "POINT"
                    or load.kind == "MOMENT"
                    or load.kind == "AXIAL_POINT"
                ):
                    b = min(self.length, load.a + 0.001)
                    c = max(0, load.a - 0.001)
                    extra_stations.extend([c, load.a, b])
                    max_stations.extend([c, load.a, b])

                elif load.kind == "LINE" or load.kind == "AXIAL_LINE":
                    c = min(self.length, load.b + 0.001)
                    d = max(0, load.a - 0.001)
                    extra_stations.extend([d, load.a, load.b, c])
                    max_stations.extend([d, load.a, load.b, c])
                else:
                    pass

            stations.extend(extra_stations)

        stations.sort()
        max_stations.sort()

        # Make sure the first and last stations do not exceed the beam

        if stations[0] < 0:
            stations[0] = 0

        if stations[-1] > self.length:
            stations[-1] = self.length
        
        if max_stations[0] < 0:
            max_stations[0] = 0

        if max_stations[-1] > self.length:
            max_stations[-1] = self.length

        # Remove duplicate locations
        self.calcstations = sorted(set(stations))
        self.maxstations = sorted((set(max_stations)))

        self._stations = True

    def generate_Loading_function(self, load_combination):
        """Generate piecewise polynomial functions representing distributed loads
    
        Creates and stores piecewise polynomial functions for loads in both the x and y
        directions by combining the load contributions from all applied loads that are
        active in the specified load combination.
    
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to generate the loading functions
        
        Notes
        -----
        The resulting functions are stored in the instance dictionaries `self.Wx` 
        and `self.Wy` using the load combination name as the key. These functions 
        represent the applied load distribution before integration into shear, 
        moment, and deflection functions.
    
        Each load's contribution is scaled by its corresponding factor from the
        load combination. Inactive loads (with factor=0) are skipped.
    
        These functions are primarily used for:
        1. Visualization of applied loads
        2. Input for generating internal force functions
        3. Integration to produce shear, moment and deflection functions
        """
        wy = loadtypes.Piecewise_Polynomial()
        wx = loadtypes.Piecewise_Polynomial()

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)

                if load_factor != 0:
                    wx = wx.combine(load.Wx, 1, load_factor)
                    wy = wy.combine(load.Wy, 1, load_factor)

        self.Wx[load_combination.name] = wx
        self.Wy[load_combination.name] = wy

    def generate_Axial_function(self, load_combination):
        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecewise functions to build the total function from the loading
        ax = loadtypes.Piecewise_Polynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], 0, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)
        fyj = loadtypes.R2_Point_Load(Fendlocal[4, 0], self.length, self)
        mzj = loadtypes.R2_Point_Moment(Fendlocal[5, 0], self.length, self)

        ax = ax.combine(fxi.Ax, 1, 1)
        ax = ax.combine(fyi.Ax, 1, 1)
        ax = ax.combine(mzi.Ax, 1, 1)
        ax = ax.combine(fxj.Ax, 1, 1)
        ax = ax.combine(fyj.Ax, 1, 1)
        ax = ax.combine(mzj.Ax, 1, 1)

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)

                if load_factor != 0:
                    ax = ax.combine(load.Ax, 1, load_factor)

        self.A[load_combination.name] = ax

    def generate_Vy_function(self, load_combination):
        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecewise functions to build the total function from the loading
        vy = loadtypes.Piecewise_Polynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], 0, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)
        fyj = loadtypes.R2_Point_Load(Fendlocal[4, 0], self.length, self)
        mzj = loadtypes.R2_Point_Moment(Fendlocal[5, 0], self.length, self)

        vy = vy.combine(fxi.Vy, 1, 1)
        vy = vy.combine(fyi.Vy, 1, 1)
        vy = vy.combine(mzi.Vy, 1, 1)
        vy = vy.combine(fxj.Vy, 1, 1)
        vy = vy.combine(fyj.Vy, 1, 1)
        vy = vy.combine(mzj.Vy, 1, 1)

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)
                if load_factor != 0:
                    vy = vy.combine(load.Vy, 1, load_factor)

        self.Vy[load_combination.name] = vy

    def generate_Mz_function(self, load_combination):
        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecewise functions to build the total function from the loading
        Mzx = loadtypes.Piecewise_Polynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], 0, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)
        fyj = loadtypes.R2_Point_Load(Fendlocal[4, 0], self.length, self)
        mzj = loadtypes.R2_Point_Moment(Fendlocal[5, 0], self.length, self)

        Mzx = Mzx.combine(fxi.Mz, 1, 1)
        Mzx = Mzx.combine(fyi.Mz, 1, 1)
        Mzx = Mzx.combine(mzi.Mz, 1, 1)
        Mzx = Mzx.combine(fxj.Mz, 1, 1)
        Mzx = Mzx.combine(fyj.Mz, 1, 1)
        Mzx = Mzx.combine(mzj.Mz, 1, 1)

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)
                if load_factor != 0:
                    Mzx = Mzx.combine(load.Mz, 1, load_factor)

        self.Mz[load_combination.name] = Mzx

    def generate_Sz_function(self, load_combination):
        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecwise functions to build the total function from the loading
        Szx = loadtypes.Piecewise_Polynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], 0, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)
        fyj = loadtypes.R2_Point_Load(Fendlocal[4, 0], self.length, self)
        mzj = loadtypes.R2_Point_Moment(Fendlocal[5, 0], self.length, self)

        Szx = Szx.combine(fxi.Sz, 1, 1)
        Szx = Szx.combine(fyi.Sz, 1, 1)
        Szx = Szx.combine(mzi.Sz, 1, 1)
        Szx = Szx.combine(fxj.Sz, 1, 1)
        Szx = Szx.combine(fyj.Sz, 1, 1)
        Szx = Szx.combine(mzj.Sz, 1, 1)

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)
                if load_factor != 0:
                    Szx = Szx.combine(load.Sz, 1, load_factor)

        self.Sz[load_combination.name] = Szx

    def generate_DxDy_function(self, load_combination):
        """
        Generate the piecewise displacement functions for the local x and y 
        axis. !!Note the nodal displacements are not included in these functions.

        :param load_combination: load combination element
        :type load_combination: _type_
        """

        if not self._stations:
            self.stations()

        empty_f = np.zeros((6, 1))

        Fendlocal = self.end_forces_local.get(load_combination.name, empty_f)

        # Empty Piecwise functions to build the total function from the loading
        dx = loadtypes.Piecewise_Polynomial()
        dy = loadtypes.Piecewise_Polynomial()

        # Create "loads" from the end forces and combine with dx and dy
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], 0, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], 0, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], 0, self)
        fxj = loadtypes.R2_Axial_Load(Fendlocal[3, 0], self.length, self)
        fyj = loadtypes.R2_Point_Load(Fendlocal[4, 0], self.length, self)
        mzj = loadtypes.R2_Point_Moment(Fendlocal[5, 0], self.length, self)

        dx = dx.combine(fxi.Dx, 1, 1)
        dy = dy.combine(fyi.Dy, 1, 1)
        dy = dy.combine(mzi.Dy, 1, 1)
        dx = dx.combine(fxj.Dx, 1, 1)
        dy = dy.combine(fyj.Dy, 1, 1)
        dy = dy.combine(mzj.Dy, 1, 1)

        # Combine Piecewise Deflection Functions of all of the loads
        if self._loaded:
            for load in self.loads:
                load_factor = load_combination.factors.get(load.loadcase, 0)

                if load_factor != 0:
                    dx = dx.combine(load.Dx, 1, load_factor)
                    dy = dy.combine(load.Dy, 1, load_factor)

        self.Dx[load_combination.name] = dx
        self.Dy[load_combination.name] = dy

    def Wxlocal_plot(self, load_combination, scale=1, ptloadscale=1):
        if not self._stations:
            self.stations()

        wx = self.Wx.get(load_combination.name, None)

        if wx is None:
            self.generate_Loading_function(load_combination)
            wx = self.Wx.get(load_combination.name, None)

        wxlocal_span = np.zeros((len(self.calcstations), 2))

        for i, x in enumerate(self.calcstations):
            w = wx.evaluate(x)

            wp = 0

            for load in self.loads:
                if load.kind == "AXIAL_POINT":
                    if load.a == x:
                        load_factor = load_combination.factors.get(load.loadcase, 0)
                        wp += load_factor * load.p

            wxlocal_span[i, 0] = x
            wxlocal_span[i, 1] = w * scale + (wp * ptloadscale)

        return wxlocal_span

    def Wxglobal_plot(self, load_combination, scale=1, ptloadscale=1):
        wxlocal_plot = self.Wxlocal_plot(
            load_combination, scale=scale, ptloadscale=ptloadscale
        )

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        wxglobal_plot = np.matmul(wxlocal_plot, R)

        return wxglobal_plot

    def Wylocal_plot(self, load_combination, scale=1, ptloadscale=1):
        if not self._stations:
            self.stations()

        wy = self.Wy.get(load_combination.name, None)

        if wy is None:
            self.generate_Loading_function(load_combination)
            wy = self.Wy.get(load_combination.name, None)

        wylocal_span = np.zeros((len(self.calcstations), 2))

        for i, x in enumerate(self.calcstations):
            w = wy.evaluate(x)

            wp = 0

            for load in self.loads:
                if load.kind == "POINT":
                    if load.a == x:
                        load_factor = load_combination.factors.get(load.loadcase, 0)
                        wp += load_factor * load.p

            wylocal_span[i, 0] = x
            wylocal_span[i, 1] = (w * scale) + (wp * ptloadscale)

        return wylocal_span

    def Wyglobal_plot(self, load_combination, scale=1, ptloadscale=1):
        wylocal_plot = self.Wylocal_plot(
            load_combination, scale=scale, ptloadscale=ptloadscale
        )

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        wyglobal_plot = np.matmul(wylocal_plot, R)

        return wyglobal_plot

    def Alocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        ax = self.A.get(load_combination.name, None)

        if ax is None:
            self.generate_Axial_function(load_combination)
            ax = self.A.get(load_combination.name, None)

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

    def Vlocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        vy = self.Vy.get(load_combination.name, None)

        if vy is None:
            self.generate_Vy_function(load_combination)
            vy = self.Vy.get(load_combination.name, None)

        vlocal_span = np.zeros((len(self.calcstations), 2))

        for i, x in enumerate(self.calcstations):
            v = vy.evaluate(x)

            vlocal_span[i, 0] = x
            vlocal_span[i, 1] = v * scale

        return vlocal_span

    def Vglobal_plot(self, load_combination, scale):
        vlocal_plot = self.Vlocal_plot(load_combination, scale=scale)

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        vglobal_plot = np.matmul(vlocal_plot, R)

        return vglobal_plot

    def Mlocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        mzx = self.Mz.get(load_combination.name, None)

        if mzx is None:
            self.generate_Mz_function(load_combination)
            mzx = self.Mz.get(load_combination.name, None)

        # Get the Roots of the shear function for the current combo
        vy = self.Vy.get(load_combination.name, None)

        if vy is None:
            self.generate_Vy_function(load_combination)
            vy = self.Vy.get(load_combination.name, None)

        shear_roots = vy.roots()
        # Generate a new station list including the roots
        stations = sorted(set(self.calcstations + shear_roots))

        mlocal_span = np.zeros((len(stations), 2))

        for i, x in enumerate(stations):
            m = mzx.evaluate(x)

            mlocal_span[i, 0] = x
            mlocal_span[i, 1] = m * scale

        return mlocal_span

    def Mglobal_plot(self, load_combination, scale):
        mlocal_plot = self.Mlocal_plot(load_combination, scale=scale)

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        mglobal_plot = np.matmul(mlocal_plot, R)

        return mglobal_plot

    def Slocal_plot(self, load_combination, scale=1):
        if not self._stations:
            self.stations()

        Szx = self.Sz.get(load_combination.name, None)

        if Szx is None:
            self.generate_Sz_function(load_combination)
            Szx = self.Sz.get(load_combination.name, None)

        slocal_span = np.zeros((len(self.calcstations), 2))
        # slope adjustment for end displacements
        Dlocal = self.Dlocal(load_combination)

        sadjust = (Dlocal[0, 4] - Dlocal[0, 1]) / self.length

        for i, x in enumerate(self.calcstations):
            s = Szx.evaluate(x)

            slocal_span[i, 0] = x
            slocal_span[i, 1] = (s + sadjust) * scale

        return slocal_span

    def Sglobal_plot(self, load_combination, scale):
        slocal_plot = self.Slocal_plot(load_combination, scale=scale)

        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        R = np.matrix([[c, s], [-s, c]])

        sglobal_plot = np.matmul(slocal_plot, R)

        return sglobal_plot

    def Dlocal_plot(self, load_combination, scale=1):
        dx = self.Dx.get(load_combination.name, None)
        dy = self.Dy.get(load_combination.name, None)

        if dx is None or dy is None:
            self.generate_DxDy_function(load_combination)
            dx = self.Dx.get(load_combination.name, None)
            dy = self.Dy.get(load_combination.name, None)

        Dlocal = self.Dlocal(load_combination)

        # Parametric Functions defining a linear relationship for deflection
        # in each axis based on the Ux and Uy nodal displacements
        Dx = lambda x: Dlocal[0, 0] + (x / self.length) * (Dlocal[0, 3] - Dlocal[0, 0])
        Dy = lambda x: Dlocal[0, 1] + (x / self.length) * (Dlocal[0, 4] - Dlocal[0, 1])

        # Get the Roots of the slope function for the current combo
        sz = self.Sz.get(load_combination.name, None)

        if sz is None:
            self.generate_Sz_function(load_combination)
            sz = self.Sz.get(load_combination.name, None)

        slope_roots = sz.roots()
        # Generate a new station list including the roots
        stations = sorted(set(self.calcstations + slope_roots))

        dlocal_span = np.zeros((len(stations), 2))

        for i, x in enumerate(stations):
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
    
    def Mzextremes(self, load_combination):
        """Find maximum and minimum bending moment values along the element
    
        Calculates the extreme moment values by:
        1. Evaluating the moment at predefined stations along the element
        2. Finding zeros of the shear force function (where moment extremes occur)
        3. Checking moment values at all critical points
    
        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to calculate extreme moments
        
        Returns
        -------
        dict
            Dictionary containing extreme moment values and their locations:
            - 'MaxM': [position, value] - Maximum positive moment
            - 'MinM': [position, value] - Maximum negative moment
        
        Notes
        -----
        This method is crucial for structural design as the extreme moment values
        are needed for member capacity checks. The zeros of the shear function
        are included as these are the locations where moments reach local extremes.
        """
        if not self._stations:
            self.stations()

        mzx = self.Mz.get(load_combination.name, None)

        if mzx is None:
            self.generate_Mz_function(load_combination)
            mzx = self.Mz.get(load_combination.name, None)

        # Get the Roots of the shear function for the current combo
        vy = self.Vy.get(load_combination.name, None)

        if vy is None:
            self.generate_Vy_function(load_combination)
            vy = self.Vy.get(load_combination.name, None)

        shear_roots = vy.roots()
        # Generate a new station list including the roots
        stations = sorted(set(self.maxstations + shear_roots))
        maxM = [0,0]
        minM = [0,0]

        for x in stations:
            m = mzx.evaluate(x)
            maxM[1] = max(maxM[1],m)
            minM[1] = min(minM[1],m)
            if maxM[1] == m:
                maxM[0] = x
            if minM[1] == m:
                minM[0] = x
        
        return {"MaxM":maxM,"MinM":minM}

    def validate_hinges(self):
        """
        Validates that hinges are correctly applied based on the node restraints.
        Raises an error if there is a conflict.
        """
        # Check i-node (start node)
        if self.hinges[0] == 1 and self.inode.restraints[2] == 1:
            raise ValueError(
                f"Conflict: Hinge applied at i-node (UID: {self.inode.uid}) of frame (UID: {self.uid}), "
                f"but the node is rotationally restrained (Rz = 1)."
            )

        # Check j-node (end node)
        if self.hinges[1] == 1 and self.jnode.restraints[2] == 1:
            raise ValueError(
                f"Conflict: Hinge applied at j-node (UID: {self.jnode.uid}) of frame (UID: {self.uid}), "
                f"but the node is rotationally restrained (Rz = 1)."
            )

    @property
    def has_distributed_loads(self):
        """Check if element has any distributed loads applied"""
        return any(load.kind == "LINE" or load.kind == "AXIAL_LINE" for load in self.loads) if self.loads else False
