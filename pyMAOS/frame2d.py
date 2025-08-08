# import os
import sys
import pint
import numpy as np

from pyMAOS.display_utils import display_node_load_vector_in_units
import pyMAOS.loading as loadtypes
from pyMAOS.elements import Element

import pyMAOS
from pyMAOS import unit_manager


# Create unit registry
# from pyMAOS.units_mod import ureg
Q_ = pyMAOS.unit_manager.ureg.Quantity

def convert_to_quantity(value, unit_str):
    """Convert a value to a quantity with units if it's not already one"""
    if isinstance(value, pint.Quantity):
        return value
    return Q_(value, unit_str)

class R2Frame(Element):
    # Class-level flag to control plotting for all instances
    plot_enabled = False

    def __init__(self, uid, inode, jnode, material, section):
        super().__init__(uid, inode, jnode, material, section)
        self.type = "FRAME"
        self.hinges = [0, 0]
        self.loads = []
        self.fixed_end_forces_local = {}
        self.fixed_end_forces_global = {}

        self.end_forces_global={}
        self.end_forces_local={}


        # Instance-level flag that can override the class setting
        self._plot_enabled = True  # None means use the class setting

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

    @property
    def plot_enabled(self):
        """Get the plot enabled state for this instance."""
        if self._plot_enabled is None:
            return R2Frame.plot_enabled
        return self._plot_enabled

    @plot_enabled.setter
    def plot_enabled(self, value):
        """Set the plot enabled state for this instance."""
        self._plot_enabled = bool(value)

    def set_plotting(self, enabled=True):
        """Enable or disable plotting for this element."""
        self._plot_enabled = bool(enabled)
        return self  # Allow chaining

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

    # def _parse_load_value(self, load_value):
    #     """
    #     Parse a load value that may be a float or string with units.
    #
    #     Parameters
    #     ----------
    #     load_value : float or str
    #         Load value, either as a number or string with units (e.g., "50kip", "100kN")
    #
    #     Returns
    #     -------
    #     float
    #         Load value in SI units (Newtons for forces, Newton-meters for moments)
    #     """
    #     # If it's already a number, return it as-is (assume SI units)
    #     if isinstance(load_value, (int, float)):
    #         return float(load_value)
    #
    #     # If it's a string, try to parse units
    #     if isinstance(load_value, str):
    #         try:
    #             # Import the parse function from units_mod
    #             from pyMAOS.units_mod import parse_value_with_units
    #             import pint
    #
    #             # Parse the load string
    #             parsed_value = parse_value_with_units(load_value)
    #
    #             # If it has units, convert to SI units
    #             if isinstance(parsed_value, pint.Quantity):
    #                 try:
    #                     # Try to convert to force units (Newtons) first
    #                     force_value = parsed_value.to('N').magnitude
    #                     return float(force_value)
    #                 except Exception:
    #                     try:
    #                         # If that fails, try moment units (Newton-meters)
    #                         moment_value = parsed_value.to('N*m').magnitude
    #                         return float(moment_value)
    #                     except Exception as e:
    #                         print(f"Warning: Could not convert '{load_value}' to SI units (N or N*m): {e}")
    #                         # Fall back to magnitude if conversion fails
    #                         return float(parsed_value.magnitude)
    #             else:
    #                 # No units, just return the numeric value
    #                 return float(parsed_value)
    #
    #         except Exception as e:
    #             print(f"Warning: Could not parse load value '{load_value}': {e}")
    #             # Try to convert directly to float as fallback
    #             try:
    #                 return float(load_value)
    #             except Exception:
    #                 raise ValueError(f"Could not parse load value: {load_value}")
    #
    #     # If we get here, something unexpected happened
    #     raise ValueError(f"Unsupported load value type: {type(load_value)}")

    # def _parse_position_value(self, position_value):
    #     """
    #     Parse a position value that may be a float or string with units.
    #
    #     Parameters
    #     ----------
    #     position_value : float or str
    #         Position value, either as a number or string with units (e.g., "10ft", "3m")
    #
    #     Returns
    #     -------
    #     float
    #         Position value in SI units (meters)
    #     """
    #     # If it's already a number, return it as-is (assume SI units)
    #     if isinstance(position_value, (int, float)):
    #         return float(position_value)
    #
    #     # If it's a string, try to parse units
    #     if isinstance(position_value, str):
    #         try:
    #             # Import the parse function from units_mod
    #             from pyMAOS.units_mod import parse_value_with_units
    #             import pint
    #
    #             # Parse the position string
    #             parsed_value = parse_value_with_units(position_value)
    #
    #             # If it has units, convert to SI units (meters)
    #             if isinstance(parsed_value, pint.Quantity):
    #                 try:
    #                     # Convert to length units (meters)
    #                     length_value = parsed_value.to('m').magnitude
    #                     return float(length_value)
    #                 except Exception as e:
    #                     print(f"Warning: Could not convert '{position_value}' to meters: {e}")
    #                     # Fall back to magnitude if conversion fails
    #                     return float(parsed_value.magnitude)
    #             else:
    #                 # No units, just return the numeric value
    #                 return float(parsed_value)
    #
    #         except Exception as e:
    #             print(f"Warning: Could not parse position value '{position_value}': {e}")
    #             # Try to convert directly to float as fallback
    #             try:
    #                 return float(position_value)
    #             except Exception:
    #                 raise ValueError(f"Could not parse position value: {position_value}")
    #
    #     # If we get here, something unexpected happened
    #     raise ValueError(f"Unsupported position value type: {type(position_value)}")

    def add_point_load(self, p: pint.Quantity, a: pint.Quantity, case="D", direction="y", location_percent=False):
        """Add a concentrated point load to the frame element
        
        Parameters
        ----------
        p : float or str
            Magnitude of the point load. Can be numeric or string with units (e.g., "50kip", "100kN")
        a : float or str
            Position along the element where the load is applied. Can be numeric or string with units (e.g., "10ft", "3m")
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
        # Parse the load magnitude with units
        # p = self._parse_load_value(p)
        #
        # # Parse the position with units (unless it's a percentage)
        # if location_percent:
        #     # For percentages, convert to decimal and multiply by length
        #     a = float(a) / 100 * self.length
        # # else:
        # #     a = self._parse_position_value(a)
            
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
            loadx=loadtypes.R2_Axial_Load(pxx, a, self, loadcase=case)
            loady=loadtypes.R2_Point_Load(pyy, a, self, loadcase=case)
            self.loads.append(loadx)
            self.loads.append(loady)
        else:
            # Load is applied in the local member axis

            if direction == "xx":
                load=loadtypes.R2_Axial_Load(p, a, self, loadcase=case)
                self.loads.append(load)
            else:
                load=loadtypes.R2_Point_Load(p, a, self, loadcase=case)
                self.loads.append(load)

            # Only plot if plotting is enabled
            if self.plot_enabled:
                fig = load.plot_all_ppoly_functions()
                fig.show()  # If you want to display immediately

        self._stations = False
        self._loaded = True

    def add_distributed_load(
        self,
        wi: pint.Quantity,
        wj: pint.Quantity,
        a: pint.Quantity,
        b: pint.Quantity,
        case: str = "D",
        direction: str = "y",
        location_percent: bool = False,
        projected: bool = False
    ):
        """Add a distributed load to the frame element
        
        Parameters
        ----------
        wi : float or str
            Starting intensity of the distributed load. Can be numeric or string with units (e.g., "0.5kip/in", "10kN/m")
        wj : float or str
            Ending intensity of the distributed load. Can be numeric or string with units (e.g., "0.5kip/in", "10kN/m")
        a : float or str
            Starting position of the distributed load. Can be numeric or string with units (e.g., "0ft", "0m")
        b : float or str
            Ending position of the distributed load. Can be numeric or string with units (e.g., "10ft", "3m")
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

        # Parse the positions with units (unless they're percentages)
        if location_percent:
            a = (float(a) / 100) * self.length
            b = (float(b) / 100) * self.length
            
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
            if abs(wxxi.magnitude) > 1e-10 or abs(wxxj.magnitude) > 1e-10:  # Only add if at least one component is non-zero
                load=loadtypes.R2_Axial_Linear_Load(wxxi, wxxj, a, b, self, loadcase=case); print(load)
                self.loads.append(load)
            if abs(wyyi.magnitude) > 1e-10 or abs(wyyj.magnitude) > 1e-10:  # Also check transverse load
                load=loadtypes.LinearLoadXY(wyyi, wyyj, a, b, self, loadcase=case); print(load)
                load.print_detailed_analysis()
                if self.plot_enabled:
                    load.plot_all_functions()
                print(load)
                self.loads.append(load)
        else:
            # Load is applied in the local member axis

            if direction == "xx":
                load=loadtypes.R2_Axial_Linear_Load(wi, wj, a, b, self, loadcase=case); print(load)
                load.print_detailed_analysis()
                self.loads.append(load)
            else:
                if projected:
                    wi = (self.jnode.x - self.inode.x) * wi / self.length
                    wj = (self.jnode.x - self.inode.x) * wj / self.length
                load = loadtypes.LinearLoadXY(wi, wj, a, b, self, loadcase=case); print(load)
                load.print_detailed_analysis()
                if self.plot_enabled:
                    load.plot_all_functions()
                self.loads.append(load)

        self._stations = False
        self._loaded = True

    def _parse_distributed_load_value(self, load_value):
        """
        Parse a distributed load value that may be a float or string with units.
        
        Parameters
        ----------
        load_value : float or str
            Distributed load value, either as a number or string with units (e.g., "0.5kip/in", "10kN/m")
            
        Returns
        -------
        float
            Distributed load value in SI units (N/m)
        """
        # If it's already a number, return it as-is (assume SI units)
        if isinstance(load_value, (int, float)):
            return float(load_value)
        
        # If it's a string, try to parse units
        if isinstance(load_value, str):
            try:
                # Import the parse function from units_mod
                from pyMAOS.pymaos_units import parse_value_with_units
                import pint
                
                # Parse the distributed load string
                parsed_value = parse_value_with_units(load_value)
                
                # If it has units, convert to SI units (N/m)
                if isinstance(parsed_value, pint.Quantity):
                    try:
                        # Convert to distributed load units (N/m)
                        distributed_load_value = parsed_value.to('N/m').magnitude
                        return float(distributed_load_value)
                    except Exception as e:
                        print(f"Warning: Could not convert '{load_value}' to N/m: {e}")
                        # Fall back to magnitude if conversion fails
                        return float(parsed_value.magnitude)
                else:
                    # No units, just return the numeric value
                    return float(parsed_value)
                    
            except Exception as e:
                print(f"Warning: Could not parse distributed load value '{load_value}': {e}")
                # Try to convert directly to float as fallback
                try:
                    return float(load_value)
                except Exception:
                    raise ValueError(f"Could not parse distributed load value: {load_value}")
        
        # If we get here, something unexpected happened
        raise ValueError(f"Unsupported distributed load value type: {type(load_value)}")

    def add_moment_load(self, m: pint.Quantity, a: pint.Quantity, case="D", location_percent=False):
        """Add a concentrated moment to the frame element
        
        Parameters
        ----------
        m : float or str
            Magnitude of the moment (positive according to right-hand rule). Can be numeric or string with units (e.g., "500kip*ft", "1000kN*m")
        a : float or str
            Position along the element where the moment is applied. Can be numeric or string with units (e.g., "5ft", "1.5m")
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
        # Parse the moment magnitude with units
        # m = self._parse_load_value(m)  # Works for both forces and moments
        #
        # # Parse the position with units (unless it's a percentage)
        # if location_percent:
        #     a = (float(a) / 100) * self.length
        # else:
        #     a = self._parse_position_value(a)
            
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
        # Initialize fixed end forces vector
        from pyMAOS import INTERNAL_FORCE_UNIT, INTERNAL_MOMENT_UNIT
        zero_force=pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_FORCE_UNIT)
        zero_moment=pyMAOS.unit_manager.ureg.Quantity(0, pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT)
        fef = np.array([zero_force,zero_force,zero_moment,zero_force,zero_force,zero_moment], dtype=object)
        # Process each load applied to the element
        print(f"Processing {len(self.loads)} loads on element {self.uid}:", file=sys.stdout)
        for load_idx, load in enumerate(self.loads):
            print(f"  Load {load_idx}: {load.kind} (case '{load.loadcase}')", file=sys.stdout)
            load_case = load.loadcase
            load_factor = load_combination.factors.get(load_case, 0)
            
            # Skip loads that don't contribute to this combination
            if load_factor == 0:
                print(f"  Load {load_idx}: {load.kind} (case '{load_case}') - skipped (factor=0)", file=sys.stdout)
                continue
                
            # Calculate FEF contribution from this load
            load_fef = load.load_fef()

            from pyMAOS.pymaos_units import array_convert_to_unit_system
            _ = array_convert_to_unit_system(load_fef, "imperial")

            factored_fef = np.array([load_factor * f for f in load_fef], dtype=object)
            # print(factored_fef)  # Shows scaled quantities with preserved units

            # print(f"Element {self.uid} load idx {load_idx}: {load.kind} (case '{load_case}', factor={load_factor})", file=sys.stdout)
            # print(f"    Raw FEF:\n{load_fef}", file=sys.stdout)
            # print(f"    Factored:\n{factored_fef}", file=sys.stdout)

            # Add to total FEF
            fef = fef + factored_fef

        from pyMAOS.pymaos_units import array_convert_to_unit_system



        # Handle hinge conditions - these modify the fixed end forces for partial releases
        if self.hinges == [1, 0]:  # Hinge at start node
            Mi = fef[2]
            L = self.length

            print(f"  Applying hinge at start node - redistributing moment Mi={Mi}", file=sys.stdout)
            fef[1] = fef[1] - ((3 / (2 * L)) * Mi)
            fef[2] = zero_moment # Zero moment at hinge
            fef[4] = fef[4] + ((3 / (2 * L)) * Mi)
            fef[5] = fef[5] - (Mi / 2)
            
        elif self.hinges == [0, 1]:  # Hinge at end node
            Mj = fef[5]
            L = self.length

            print(f"  Applying hinge at end node - redistributing moment Mj={Mj}", file=sys.stdout)
            fef[1] = fef[1] - ((3 / (2 * L)) * Mj)
            fef[2] = fef[2] - (Mj / 2)
            fef[4] = fef[4] + ((3 / (2 * L)) * Mj)
            fef[5] = zero_moment  # Zero moment at hinge
            
        elif self.hinges == [1, 1]:  # Hinges at both nodes
            Mi = fef[2]
            Mj = fef[5]
            L = self.length

            print(f"  Applying hinges at both nodes - redistributing moments Mi={Mi}, Mj={Mj}", file=sys.stdout)
            fef[1] = fef[1] - ((Mj + Mi) / L)
            fef[2] = zero_moment  # Zero moment at hinge
            fef[4] = fef[4] + ((Mj + Mi) / L)
            fef[5] = zero_moment  # Zero moment at hinge

        print(f"  Final FEF for element {self.uid}:\n{fef}", file=sys.stdout)
        # display_node_load_vector_in_units(fef[0:3], self.inode.uid,
        #                                   force_unit=self.structure.units['force'],
        #                                   length_unit=self.structure.units['distance'],
        #                                   load_combo_name=None)
        # display_node_load_vector_in_units(fef[3:6], self.jnode.uid,
        #                                   force_unit=self.structure.units['force'],
        #                                   length_unit=self.structure.units['distance'],
        #                                   load_combo_name=None)
        return fef

    def FEFglobal(self, load_combination):
        """
        Transform fixed end forces from local to global coordinates.
        """
        # Get fixed end forces in local coordinates
        local_fef = self.FEF(load_combination)
        # fef = np.transpose(fef)
        # print(f"DEBUG: FEF in local coordinates: {fef}")

        # Get transformation matrix
        rotation_matrix = self.set_rotation_matrix()

        # Check if we're dealing with quantities with units
        if isinstance(local_fef[0], pint.Quantity):
            # Store units for each component
            fef_units = [f.units for f in local_fef]

            # Extract magnitudes for calculation
            fef_magnitudes = np.array([f.magnitude for f in local_fef], dtype=np.float64)

            # Perform the transformation with magnitudes only
            import scipy.linalg as sla
            elem_global_fef_magnitudes = sla.blas.dgemv(1.0, rotation_matrix.T, fef_magnitudes); print(elem_global_fef_magnitudes)
            # print(f"DEBUG: Using scipy.linalg.blas: {result_magnitudes.shape}")
            # print(f"DEBUG: Result magnitudes: {result_magnitudes}")
            # Reattach original units
            elem_global_fef = np.array([unit_manager.ureg.Quantity(mag, unit)
                              for mag, unit in zip(elem_global_fef_magnitudes, fef_units)],
                             dtype=object)
            from pymaos_units import array_convert_to_unit_system
            print(f"FEFglobal for element {self.uid}:"); _ = array_convert_to_unit_system(elem_global_fef, "imperial")

        else:
            # If no units, proceed with standard matrix multiplication
            elem_global_fef=np.matmul(np.transpose(rotation_matrix), local_fef)

        self.fixed_end_forces_global[load_combination.name]=elem_global_fef
        return elem_global_fef

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
        E: pint.Quantity = self.material.E
        Ixx = self.section.Ixx
        A = self.section.Area
        L = self.length

        # Initialize matrix with zeros
        k = np.zeros((6, 6), dtype=object)

        # Common terms
        AE_L = A * E / L

        # Axial terms (common to all hinge configurations)
        k[0, 0] = AE_L
        k[3, 3] = AE_L
        k[0, 3] = -AE_L
        k[3, 0] = -AE_L

        EI = E * Ixx
        EI_L = EI / L
        EI_L2 = EI_L / L
        EI_L3 = EI_L2 / L

        # Apply appropriate bending terms based on hinge configuration
        if self.hinges == [1, 1]:  # Both ends hinged - only axial stiffness
            pass  # No additional terms needed

        elif self.hinges == [1, 0]:  # Hinge at i-end
            k[1, 1] = 3 * EI_L3
            k[4, 4] = 3 * EI_L3
            k[1, 4] = -3 * EI_L3
            k[4, 1] = -3 * EI_L3
            k[1, 5] = 3 * EI_L2
            k[5, 1] = 3 * EI_L2
            k[4, 5] = -3 * EI_L2
            k[5, 4] = -3 * EI_L2
            k[5, 5] = 3 * EI_L

        elif self.hinges == [0, 1]:  # Hinge at j-end
            k[1, 1] = 3 * EI_L3
            k[4, 4] = 3 * EI_L3
            k[1, 2] = 3 * EI_L2
            k[2, 1] = 3 * EI_L2
            k[1, 4] = -3 * EI_L3
            k[4, 1] = -3 * EI_L3
            k[2, 4] = -3 * EI_L2
            k[4, 2] = -3 * EI_L2
            k[2, 2] = 3 * EI_L

        else:  # No hinges - standard beam element
            k[1, 1] = 12 * EI_L3
            k[4, 4] = 12 * EI_L3
            k[1, 2] = 6 * EI_L2
            k[2, 1] = 6 * EI_L2
            k[1, 4] = -12 * EI_L3
            k[4, 1] = -12 * EI_L3
            k[1, 5] = 6 * EI_L2
            k[5, 1] = 6 * EI_L2
            k[2, 2] = 4 * EI_L
            k[2, 4] = -6 * EI_L2
            k[4, 2] = -6 * EI_L2
            k[2, 5] = 2 * EI_L
            k[5, 2] = 2 * EI_L
            k[4, 5] = -6 * EI_L2
            k[5, 4] = -6 * EI_L2
            k[5, 5] = 4 * EI_L

        print(f"Local stiffness matrix for element {self.uid} with hinges {self.hinges}:",)
        # local_stiffness_matrix = np.matrix(k)
        # print(f"Local stiffness matrix for element {self.uid}:\n{local_stiffness_matrix}")

        # # First try to get units from element's structure
        # units_dict = None
        #
        # # Option 1: Check if element has direct reference to structure with units
        # if hasattr(self, 'structure') and hasattr(self.structure, 'units'):
        #     units_dict = self.structure.units
        # else:            # Option 2: Use unit manager to get current units
        #     
        #     units_dict = unit_manager.get_current_units()
        # # Get current unit system directly from the manager
        #
        # print(f"Local stiffness matrix for element {self.uid}:{self.k_with_units()}\n")
        #
        # # self.display_stiffness_matrix_in_units(local_stiffness_matrix, units_dict)

        return k

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
        Dlocal = self.set_displacement_local(load_combination.name)
        Qf = np.reshape(self.FEF(load_combination), (-1, 1))
        k = self.k()
        k_with_units=self.k_with_units()
        print(f"Local stiffness matrix for element {self.uid} under load combination '{load_combination.name}':\n{k_with_units}")
        FL = np.matmul(self.k(), Dlocal.T)

        self.end_forces_local[load_combination.name] = FL + Qf

    def set_end_forces_global(self, load_combination):
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

        print(f"Calculating global displacements for element {self.uid} under load combination '{load_combination.name}'")
        Dglobal = self.set_displacement_global(load_combination.name)

        # global stiffness matrix
        KG = self.kglobal()

        # Replace np.matmul with scipy.linalg.blas.dgemv for proper vector handling
        import scipy.linalg as sla

        # Check if we're dealing with quantities with units
        if True or isinstance(Dglobal[0], pint.Quantity):
            # Store units for calculation
            from pyMAOS.quantity_utils import numpy_array_of_quantity_to_numpy_array_of_float64, extract_units_from_quantities
            dglobal_units = extract_units_from_quantities(Dglobal)

            # Extract magnitudes for calculation
            kg_magnitudes = numpy_array_of_quantity_to_numpy_array_of_float64(KG)
            dglobal_magnitudes = numpy_array_of_quantity_to_numpy_array_of_float64(Dglobal)

            # Perform matrix-vector multiplication with SciPy BLAS
            result_magnitudes = sla.blas.dgemv(1.0, kg_magnitudes, dglobal_magnitudes)
            print(f"DEBUG: Using scipy.linalg.blas for matrix-vector multiplication: shape={result_magnitudes.shape}")

            dglobal_units_conjugate = unit_manager.get_conjugate_units_array(dglobal_units)
            # Reattach units to result
            tmp_list = [unit_manager.ureg.Quantity(mag, unit)
                        for mag, unit in zip(result_magnitudes, dglobal_units_conjugate)]
            FG = np.array(tmp_list, dtype=object)
        else:
            # For non-quantity arrays, use scipy.linalg.blas directly
            FG = sla.blas.dgemv(1.0, KG, Dglobal)

        print(f"Global end forces for element {self.uid} under load combination '{load_combination.name}':\n{FG}")

        # Store the global end forces
        # if not hasattr(self, 'end_forces_global'):
        #     self.end_forces_global = {}
        # print("FG:", FG, sep="\n")

        if not load_combination.name in self.fixed_end_forces_global.keys():
            fefg = self.FEFglobal(load_combination)
        else:
            fefg=self.fixed_end_forces_global[load_combination.name]
        print("Qfg:", fefg, sep="\n")
        print(f"Global fixed end forces for element {self.uid} under load combination '{load_combination.name}':\n{fefg}")

        # Combine global end forces with fixed end forces

        from pyMAOS.quantity_utils import add_arrays_with_units

        # Element-wise addition with proper unit handling
        ret_val = add_arrays_with_units(FG, fefg)
        print(f"DEBUG: element {self.uid} ret_val={ret_val}")
        self.end_forces_global[load_combination.name] = ret_val

        # print(f"End forces in local coordinates for element {self.uid} under load combination '{load_combination.name}':\n{self.end_forces_local.get(load_combination.name, 'Not calculated')}")
        # self.Flocal(load_combination)

        return self.end_forces_global[load_combination.name]

    def stations(self, num_stations=10):
        """
        Define evenly distributed points along the member to compute internal
        actions. Additional points are generated for load application points.

        This also generates a reduced set of points for use in the max/min
        internal action functions.

        Parameters
        ----------
        num_stations : int, optional
            Number of equally spaced points along the member, defaults to 10
        """
        # Convert to magnitude if length is a quantity
        length_value = self.length.magnitude if hasattr(self.length, 'magnitude') else self.length

        # Parametric list of stations between 0 and 1
        eta = [0 + i * (1 / num_stations) for i in range(num_stations + 1)]

        stations = [length_value * i for i in eta]
        max_stations = [0, length_value]

        if self._loaded:
            extra_stations = []

            for load in self.loads:
                # Handle different load types correctly
                if hasattr(load, 'a'):
                    # Position of the load - convert to magnitude if it's a Quantity
                    a = load.a.magnitude if hasattr(load.a, 'magnitude') else load.a
                    extra_stations.append(a)

                # For distributed loads with 'b' attribute (end position)
                if hasattr(load, 'b'):
                    b = load.b.magnitude if hasattr(load.b, 'magnitude') else load.b
                    c = min(length_value, b + 0.001)
                    extra_stations.append(c)

            stations.extend(extra_stations)

        stations.sort()
        max_stations.sort()

        # Make sure the first and last stations do not exceed the beam
        if stations[0] < 0:
            stations[0] = 0

        if stations[-1] > length_value:
            stations[-1] = length_value

        if max_stations[0] < 0:
            max_stations[0] = 0

        if max_stations[-1] > length_value:
            max_stations[-1] = length_value

        # Remove duplicate locations
        self.calcstations = sorted(set(stations))
        self.maxstations = sorted(set(max_stations))

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
        wy = loadtypes.PiecewisePolynomial()
        wx = loadtypes.PiecewisePolynomial()

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
        ax = loadtypes.PiecewisePolynomial()

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
        vy = loadtypes.PiecewisePolynomial()

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
        Mzx = loadtypes.PiecewisePolynomial()

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
        Szx = loadtypes.PiecewisePolynomial()

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
        print(Fendlocal)
        # Empty Piecwise functions to build the total function from the loading
        dx = loadtypes.PiecewisePolynomial()
        dy = loadtypes.PiecewisePolynomial()

        # Create "loads" from the end forces and combine with dx and dy
        zero_length=unit_manager.ureg.Quantity(0, unit_manager.INTERNAL_LENGTH_UNIT)
        fxi = loadtypes.R2_Axial_Load(Fendlocal[0, 0], zero_length, self)
        fyi = loadtypes.R2_Point_Load(Fendlocal[1, 0], zero_length, self)
        mzi = loadtypes.R2_Point_Moment(Fendlocal[2, 0], zero_length, self)
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
        Dlocal = self.set_displacement_local(load_combination)

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

    def dlocal_span(self, load_combination, scale=1):
        """
        Calculate displacement function in local coordinates along the member

        Parameters
        ----------
        load_combination : LoadCombo
            Load combination to use for displacement calculations
        scale : float, optional
            Scale factor for visual amplification of displacements

        Returns
        -------
        numpy.ndarray
            Array of points representing the displaced shape in local coordinates
        """
        dx = self.Dx.get(load_combination.name, None)
        dy = self.Dy.get(load_combination.name, None)

        if dx is None or dy is None:
            self.generate_DxDy_function(load_combination)
            dx = self.Dx.get(load_combination.name, None)
            dy = self.Dy.get(load_combination.name, None)

        Dlocal = self.set_displacement_local(load_combination)

        # Get length as a scalar value for calculations
        length_value = self.length.magnitude if hasattr(self.length, 'magnitude') else self.length

        # Parametric Functions defining a linear relationship for deflection
        # in each axis based on the Ux and Uy nodal displacements
        def Dx(x):
            return Dlocal[0, 0] + (x / length_value) * (Dlocal[0, 3] - Dlocal[0, 0])

        def Dy(x):
            return Dlocal[0, 1] + (x / length_value) * (Dlocal[0, 4] - Dlocal[0, 1])

        # Get the Roots of the slope function for the current combo
        sz = self.Sz.get(load_combination.name, None)

        if sz is None:
            self.generate_Sz_function(load_combination)
            sz = self.Sz.get(load_combination.name, None)

        slope_roots = sz.roots()
        # Generate a new station list including the roots
        stations = sorted(set(self.calcstations + slope_roots))

        dlocal_span = np.zeros((len(stations), 2), dtype=object)

        for i, x in enumerate(stations):
            dxl = dx.evaluate(x) + Dx(x)
            dyl = dy.evaluate(x) + Dy(x)

            dlocal_span[i, 0] = x + (dxl * scale)
            dlocal_span[i, 1] = dyl * scale

        return dlocal_span

    def dglobal_span(self, load_combination, scale=1):
        """
        Calculate displacement function in global coordinates along the member

        Parameters
        ----------
        load_combination : LoadCombo
            Load combination to use for displacement calculations
        scale : float, optional
            Scale factor for visual amplification of displacements

        Returns
        -------
        numpy.ndarray
            Array of points representing the displaced shape in global coordinates
        """
        dlocal_plot = self.dlocal_span(load_combination, scale=scale)

        # Extract transformation matrix components
        c = (self.jnode.x - self.inode.x) / self.length
        s = (self.jnode.y - self.inode.y) / self.length

        # Handle magnitude if quantities with units
        if hasattr(c, 'magnitude'):
            c = c.magnitude
        if hasattr(s, 'magnitude'):
            s = s.magnitude

        print(self.rotation_matrix[0:1,0:1])
        R = np.matrix([[c, s], [-s, c]])
        print(R)
        # Transform local coordinates to global
        dglobal_plot = np.matmul(dlocal_plot, R)

        # Add global position offset (start node coordinates)
        origin_x = self.inode.x.magnitude if hasattr(self.inode.x, 'magnitude') else self.inode.x
        origin_y = self.inode.y.magnitude if hasattr(self.inode.y, 'magnitude') else self.inode.y

        for i in range(dglobal_plot.shape[0]):
            dglobal_plot[i, 0] += origin_x
            dglobal_plot[i, 1] += origin_y

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

        # Get the Roots of the shear force function for the current combo
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

def print_member_loads_to_file(member, load_combo, filename):
    """
    Print all piecewise loading functions for a member under a specific load combination to a file.

    Parameters
    ----------
    member : R2Frame
        The frame element to analyze
    load_combo : str
        Name of the load combination
    filename : str
        Path to the output file
    """
    import sys
    import os

    # Check if member has any loads
    if not hasattr(member, 'loads') or not member.loads:
        print(f"No loads defined for member {member.uid}")
        return

    # Filter loads for this load combination
    combo_loads = [load for load in member.loads if load.loadcase == load_combo.name]

    if not combo_loads:
        print(f"No loads found for load combination '{load_combo}' on member {member.uid}")
        return

    print(f"Writing load analysis for member {member.uid}, load combo '{load_combo}' to '{filename}'")

    # Open file and redirect stdout to capture the ASCII output
    original_stdout = sys.stdout

    try:
        with open(filename, 'w') as f:
            sys.stdout = f

            # Write header information
            print(f"===== MEMBER {member.uid} LOADS UNDER LOAD COMBINATION '{load_combo}' =====")
            print(f"Member length: {member.length}")
            print(f"Member section: {member.section.name if hasattr(member.section, 'name') else str(member.section)}")
            print("\n")

            # For each load, print its detailed analysis
            for i, load in enumerate(combo_loads):
                print(f"\n{'='*80}")
                print(f"LOAD {i+1}: {load}")
                print(f"{'='*80}\n")

                # Use the existing detailed analysis printing function
                load.print_detailed_analysis(num_points=20, chart_width=80, chart_height=20)
                print("\n")

            print("\nEnd of load analysis.")

    except Exception as e:
        print(f"Error writing to file: {e}")
        import traceback
        traceback.print_exc(file = sys.stderr)
    finally:
        # Restore stdout
        sys.stdout = original_stdout

    print(f"Analysis complete. Output written to '{os.path.basename(filename)}'")