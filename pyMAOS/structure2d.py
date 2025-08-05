import os
import sys

import numpy as np
import pint
from pprint import pprint
import pyMAOS
from pyMAOS import pymaos_units, unit_manager
from pyMAOS.pymaos_units import array_convert_to_unit_system
from pyMAOS.display_utils import display_node_load_vector_in_units, display_node_displacement_in_units
from quantity_utils import extract_units_from_quantities


np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal', linewidth=999)

import ast
import operator
import importlib


# # Add custom formatters with explicit type signatures
def format_with_dots(x) -> str:
    return '.'.center(12) if abs(x) < 1e-10 else f"{x:12.4g}"


def format_double(x) -> str:
    return '.'.center(13) if abs(x) < 1e-10 else f"{x:13.8g}"  # More precision for doubles


# Now use these type-annotated functions in the formatter dictionary
np.set_printoptions(precision=4,
                    suppress=False,
                    formatter={
                        'float': format_with_dots,  # For float32 and generic floats
                        'float_kind': format_with_dots,  # For all floating point types
                        'float64': format_double  # Specifically for float64 (double precision)
                    },  # type: ignore
                    linewidth=120  # Wider output to prevent unnecessary wrapping
                    )


# Parameters you can adjust:
# *	precision: Number of decimal places (4 is a good default)
# *	suppress: When True, very small values near zero are displayed as 0
# *	floatmode: Controls the display format
# *	'maxprec_equal': Best option for automatic switching between formats
# *	'fixed': Always use fixed-point notation
# *	'scientific': Always use scientific notation
# *	'unique': Use minimum digits to distinguish values

class R2Structure:
    def __init__(self, nodes, members, units=None):
        self.nodes = nodes
        self.members = members
        self.units = units or pyMAOS.unit_manager.get_current_units()  # Use unit manager as fallback

        # Validate node UIDs are unique
        self._validate_node_uids()

        # Validate member UIDs are unique
        self._validate_member_uids()

        self.create_uid_maps()  # Replace create_uid_map() with the new method

        # Validate that all member nodes exist in the node list
        for member in members:
            if member.inode.uid not in self.uid_to_index:
                raise ValueError(
                    f"Member {member.uid} references node {member.inode.uid} which doesn't exist in the structure")
            if member.jnode.uid not in self.uid_to_index:
                raise ValueError(
                    f"Member {member.uid} references node {member.jnode.uid} which doesn't exist in the structure")

        # Rest of initialization remains the same...
        self.members = members

        # Structure Type
        # 2D Structure


        # Number of Joints
        self.NJ = len(self.nodes)

        # Number of Members
        self.NM = len(self.members)

        # Number of Restraints
        self.NR = sum([sum(node.restraints) for node in self.nodes])

        # Degrees of Freedom
        self.DIM=sum([node.NJD for node in self.nodes])
        self.NDOF = self.DIM - self.NR

        # Data Stores
        self._springNodes = None
        self._nonlinearNodes = None
        self._D = {}  # Structure Displacement Vector Dictionary
        self.U=[]
        # Flags
        self._unstable = False
        self._Kgenerated = False
        self._ERRORS = []

        # Register members with the structure for unit access
        for member in self.members:
            if hasattr(member, 'set_structure'):
                member.set_structure(self)

    def set_node_uids(self):
        i = 1

        for node in self.nodes:
            node.uid = i
            i += 1

    def set_member_uids(self):
        i = 1

        for member in self.members:
            member.uid = i
            i += 1

    def _validate_node_uids(self):
        """
        Validate that all node UIDs in the structure are unique.
        Raises ValueError if duplicate UIDs are found.
        """
        uids = [node.uid for node in self.nodes]
        duplicates = set([uid for uid in uids if uids.count(uid) > 1])

        if duplicates:
            duplicate_str = ", ".join(str(uid) for uid in duplicates)
            raise ValueError(f"Duplicate node UIDs found: {duplicate_str}")

    def _validate_member_uids(self):
        """
        Validate that all member UIDs in the structure are unique.
        Raises ValueError if duplicate UIDs are found.
        """
        uids = [member.uid for member in self.members]
        duplicates = set([uid for uid in uids if uids.count(uid) > 1])

        if duplicates:
            duplicate_str = ", ".join(str(uid) for uid in duplicates)
            raise ValueError(f"Duplicate member UIDs found: {duplicate_str}")

    def spring_nodes(self):
        # loop through nodes and create a list of the nodes with springs
        # assigned to a DOF.
        springNodes = []
        nonlinearNodes = []
        for node in self.nodes:
            if node._isSpring is True:
                springNodes.append(node)

            if node._isNonLinear is True:
                nonlinearNodes.append(node)

        if springNodes:
            self._springNodes = springNodes

        if nonlinearNodes:
            self._nonlinearNodes = nonlinearNodes

    def create_uid_map(self):
        """Create a mapping from UIDs to positions"""
        self.uid_to_index = {node.uid: i for i, node in enumerate(self.nodes)}

    def create_uid_maps(self):
        """Create mappings between UIDs and positions (in both directions)"""
        # Map from UID to position index
        self.uid_to_index = {node.uid: i for i, node in enumerate(self.nodes)}

        # Map from position index to UID
        self.index_to_uid = {i: node.uid for i, node in enumerate(self.nodes)}

    def set_freedom_map(self):
        # Freedom Map
        self.FM = np.zeros(self.DIM, dtype=np.int32)  # Ensure FM is initialized as an integer array

        # Loop through the nodes mapping free and restrained joint displacements to
        # the Freedom Map (FM). This will facilitate generating the global stiffness
        # matrix in partitioned form.

        j = 0  # starting index for the first free displacement
        k = self.NDOF  # starting index for the first restraint

        for node_index, node in enumerate(self.nodes):
            for r, restraint in enumerate(node.restraints):
                fmindex = node_index * node.NJD + r

                if restraint == 0:
                    self.FM[fmindex] = j
                    j += 1
                else:
                    self.FM[fmindex] = k
                    k += 1

        return self.FM.astype(np.int32)

    def get_freedom_map(self):
        return self.FM.astype(np.int32)  # Ensure FM is returned as an integer array

    def Kstructure(self, **kwargs):
        """
        Build the structure stiffness matrix organized into paritioned form
        using the freedom map to reposition nodal DOFs

        Returns
        -------
        KSTRUCT: Numpy Matrix
            Structure Stiffness Matrix.

        """
        verbose = kwargs.get('verbose', False)
        output_dir = kwargs.get('output_dir', '.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Structure Stiffness Matrix
        KSTRUCT = np.zeros([self.DIM, self.DIM], dtype=object)
        print("KSTRUCT shape:", KSTRUCT.shape, flush=True)

        for member in self.members:
            # Freedom map for i and j nodes
            inode=member.inode
            imap = [int(self.FM[(self.uid_to_index[inode.uid]) * inode.NJD + r]) for r in range(inode.NJD)]
            jnode=member.jnode
            imap.extend([int(self.FM[(self.uid_to_index[jnode.uid]) * jnode.NJD + r]) for r in range(jnode.NJD)])
            print(f"Freedom Map Indices for Member {member.uid}:\n{imap}", flush=True)

            # Member global stiffness matrix
            kmglobal = member.kglobal()
            print(f"Member {member.uid}:\tkmglobal (Internal Units)\n{kmglobal}\n", flush=True)

            # Check if the member is a truss
            if member.type == "TRUSS":
                # For truss elements, verify that all bending/shear terms are zero
                bending_terms = [
                    kmglobal[2, 2],  # i-node bending terms
                    kmglobal[5, 5]  # j-node bending terms
                ]

                # Check if any of these terms are non-zero
                if not all(abs(term) < 1e-10 for term in bending_terms):
                    non_zero_terms = [f"[{i}]={val}" for i, val in enumerate(bending_terms) if abs(val) > 1e-10]
                    error_msg = (f"Error: Member {member.uid} is a TRUSS but has non-zero bending terms: "
                                 f"{', '.join(non_zero_terms)}. Truss elements must have zero bending stiffness.")
                    print(error_msg)
                    raise ValueError(error_msg)

            # Save each member's global stiffness matrix to a separate CSV file
            filename = os.path.join(output_dir, f'member_{member.uid}_global_stiffness_SI.csv')
            np.savetxt(filename, kmglobal, delimiter=',', fmt='%g')
            print(f"Saved member {member.uid} global stiffness matrix to {filename}", flush=True)

            self.NJD=3 # temporary fix
            for i in range(self.NJD):
                for j in range(self.NJD):
                    # print(f"i: {i}, j: {j}, imap[i]: {imap[i]}, imap[j]: {imap[j]}", flush=True)
                    # Get current values at the target positions
                    pos_ii = (imap[i], imap[j])
                    pos_ji = (imap[i + self.NJD], imap[j])
                    pos_ij = (imap[i], imap[j + self.NJD])
                    pos_jj = (imap[i + self.NJD], imap[j + self.NJD])

                    # Get source values to add
                    val_ii = kmglobal[i, j]
                    val_ji = kmglobal[i + self.NJD, j]
                    val_ij = kmglobal[i, j + self.NJD]
                    val_jj = kmglobal[i + self.NJD, j + self.NJD]

                    from pyMAOS.quantity_utils import increment_with_units
                    # Use increment_with_units to add values with unit checking
                    print(f"DEBUG: Adding to KSTRUCT at {pos_ii}: {val_ii}")
                    KSTRUCT[pos_ii] = increment_with_units(KSTRUCT[pos_ii], val_ii)

                    print(f"DEBUG: Adding to KSTRUCT at {pos_ji}: {val_ji}")
                    KSTRUCT[pos_ji] = increment_with_units(KSTRUCT[pos_ji], val_ji)

                    print(f"DEBUG: Adding to KSTRUCT at {pos_ij}: {val_ij}")
                    KSTRUCT[pos_ij] = increment_with_units(KSTRUCT[pos_ij], val_ij)

                    print(f"DEBUG: Adding to KSTRUCT at {pos_jj}: {val_jj}")
                    KSTRUCT[pos_jj] = increment_with_units(KSTRUCT[pos_jj], val_jj)


        # Loop through Spring Nodes and add the spring stiffness
        if self._springNodes:
            for node in self._springNodes:
                node_index = self.uid_to_index[node.uid]
                uxposition = int(FM[(node_index) * self.NJD + 0])
                uyposition = int(FM[(node_index) * self.NJD + 1])
                rzposition = int(FM[(node_index) * self.NJD + 2])
                kux = node._spring_stiffness[0]
                kuy = node._spring_stiffness[1]
                krz = node._spring_stiffness[2]

                # KSTRUCT[uxposition, uxposition] += kux
                # KSTRUCT[uyposition, uyposition] += kuy
                # KSTRUCT[rzposition, rzposition] += krz

                KSTRUCT.incremental_add_with_units((uxposition, uxposition), kux)
                KSTRUCT.incremental_add_with_units((uyposition, uyposition), kuy)
                KSTRUCT.incremental_add_with_units((rzposition, rzposition), krz)
        self._Kgenerated = True

        if verbose:
            np.set_printoptions(precision=4,
                                suppress=False,
                                formatter={
                                    'float': format_with_dots,  # For float32 and generic floats
                                    'float_kind': format_with_dots,  # For all floating point types
                                    'float64': format_double  # Specifically for float64 (double precision)
                                },  # type: ignore
                                linewidth=120  # Wider output to prevent unnecessary wrapping
                                )
            print("KSTRUCT:", KSTRUCT, sep="\n");
            print(KSTRUCT.shape);
            from pyMAOS.quantity_utils import print_units_matrix
            print_units_matrix(KSTRUCT)

            KSTRUCT_csv = os.path.join(output_dir, 'KSTRUCT.csv')
            np.savetxt(KSTRUCT_csv, KSTRUCT, delimiter=',', fmt='%lg')
            print(f"Saved KSTRUCT to {KSTRUCT_csv}")

        return KSTRUCT


    def nodal_force_vector(self, FM, load_combination):
        """
        Build the structure nodal force vector mapped to the same partitions
        as KSTRUCT using the freedom map (FM).

        Returns
        -------
        FG : ndarray
            Structure Nodal Force Vector with proper units.
        """
        # from pyMAOS.units_mod import INTERNAL_FORCE_UNIT, INTERNAL_MOMENT_UNIT
        # zero_force=unit_manager.get_zero_quantity(INTERNAL_FORCE_UNIT)
        # zero_moment= unit_manager.get_zero_quantity(INTERNAL_MOMENT_UNIT)

        FG = np.zeros(self.NJD * self.NJ, dtype=object)

        print(f"DEBUG: Initialized FG as QuantityArray, shape={FG.shape}")

        # Apply loads from each node
        for node_index, node in enumerate(self.nodes):
            for load_case, load in node.loads.items():
                # Display load information for debugging
                from pyMAOS.display_utils import display_node_load_vector_in_units
                display_node_load_vector_in_units(
                    load_vector=load,
                    node_uid=node.uid,
                    force_unit=self.units.get('force', 'N'),
                    length_unit=self.units.get('length', 'm'),
                    load_combo_name=load_combination.name
                )

                load_factor = load_combination.factors.get(load_case, 0)

                # Use list comprehension for pint quantities
                factored_load = [load_factor * f for f in load]
                print(f"DEBUG: Node {node.uid}, load_case={load_case}, factor={load_factor}, factored_load={factored_load}")

                for i, f in enumerate(factored_load):
                    fm_index = node_index * self.NJD + i
                    mapped_index = int(self.FM[fm_index])  # Ensure integer index

                    # Add load component using QuantityArray's add_with_units method
                    FG.incremental_add_with_units((mapped_index,), f)
                    print(f"DEBUG: Added {f} at position {mapped_index}")

        print("Nodal Force Vector:", sep="\n")
        from pyMAOS.pymaos_units import array_convert_to_unit_system
        _ = array_convert_to_unit_system(FG, "imperial")

        return FG



    def assemble_fixed_end_force(self, load_combination):
        # Initialize array with object dtype to store Quantity objects
        structure_fef = np.zeros(self.NJD * self.NJ, dtype='object')

        print(f"DEBUG: Initial PF dtype = {structure_fef.dtype}")

        for member in self.members:
            if member.type != "TRUSS":
                # Get fixed end forces in global coordinates
                elem_fef_global = member.FEFglobal(load_combination)
                # print("DEBUG: Ff type:", type(Ff), "shape:", np.shape(Ff))
                from pyMAOS.pymaos_units import array_convert_to_unit_system;
                # print(f"Member {member.uid} fixed end forces before conversion: {Ff.view(QuantityArray)}")
                _ = array_convert_to_unit_system(elem_fef_global, "imperial")

                i_index = self.uid_to_index[member.inode.uid]
                j_index = self.uid_to_index[member.jnode.uid]

                # Debug info about the returned forces
                # print(f"DEBUG: Member {member.uid} forces type: {type(Ff)} shape: {np.shape(Ff)}")
                # print(f"DEBUG: First element type: {type(Ff[0])}")

                # Add the forces directly - with object dtype, this should work with Quantity objects
                imap = self.FM[i_index * self.NJD:(i_index + 1) * self.NJD]
                print(f"DEBUG: imap for i_index {i_index}: {imap}")
                from pyMAOS.quantity_utils import increment_with_units
                # structure_fef[imap] += Ff[0:self.NJD]
                print(elem_fef_global[0:self.NJD])
                structure_fef[imap]= increment_with_units(structure_fef[imap], elem_fef_global[0:self.NJD])
                # target_indices = self.FM[j_index * self.NJD:(j_index + 1) * self.NJD]
                # source_values = Ff[self.NJD:2 * self.NJD]
                #
                # # Loop through each index and use increment_with_units
                # for i, idx in enumerate(target_indices):
                #     structure_fef[idx] = increment_with_units(structure_fef[idx], source_values[i])

                # Check units consistency after addition
                # sample_idx = FM[i_index * self.NJD]
                # print(f"DEBUG: structure_fef[{sample_idx}] type after addition: {type(structure_fef[sample_idx])}")

        from pyMAOS.quantity_utils import print_units_matrix

        print(f"Member {member.uid} Fixed End Force Vector PF:")
        print_units_matrix(structure_fef)

        # At the end, you can extract magnitudes if needed for further calculations
        # PF_magnitudes = np.array([f.magnitude if hasattr(f, 'magnitude') else f for f in structure_fef])
        # print(f"DEBUG: Final PF_magnitudes dtype = {PF_magnitudes.dtype}")
        self.structure_fef = structure_fef
        print("Structure Fixed End Force Vector structure_fef:", structure_fef, sep="\n")
        print("Put back in order by code numers:", structure_fef[self.FM], sep="\n")
        return structure_fef


    def solve_linear_static(self, load_combination, **kwargs):
        """
        Perform a linear static solution of the model using the Kff
        and FGf paritions

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Extract arguments from kwargs
        verbose = kwargs.get('verbose', False)
        # if verbose:
        #     print("--- Running in Verbose Mode ---")
        structure_state_bin = kwargs.get("structure_state_bin", None)

        # Generate Freedom Map
        self.FM = self.set_freedom_map(); print("Freedom Map:", self.FM, sep="\n")

        # Generate Full Structure Stiffness Matrix
        self.KSTRUCT = self.Kstructure(**kwargs)
        from pyMAOS.quantity_utils import numpy_array_of_quantity_to_numpy_array_of_float64


        self._verify_stable(self.FM, self.KSTRUCT)

        if self._unstable:
            raise ValueError("Structure is unstable");
            sys.exit(1)

        # Build Nodal Force Vector
        self.FG = self.nodal_force_vector(self.FM, load_combination)
        # print("Nodal Force Vector FG:\n", self.FG); print(self.FG.shape)
        from pyMAOS.pymaos_units import array_convert_to_unit_system; print("Structure nodal global forces")
        _ = array_convert_to_unit_system(self.FG, "imperial")
        # Build Member Fixed-end-Force vector
        self.structure_fef = self.assemble_fixed_end_force(load_combination)
        # print("Structure Fixed End Force Vector:\n", structure_fef); print(structure_fef.shape)

        from pymaos_units import array_convert_to_unit_system; print("Structure Fixed-end Forces")
        _ = array_convert_to_unit_system(self.structure_fef, "imperial")

        self.Kff = self.KSTRUCT[0: self.NDOF, 0: self.NDOF]; print("Kff Partition:", self.Kff, sep="\n");
        print(self.Kff.shape)  # display_stiffness_matrix_in_units(self.Kff)
        # Slice out the FGf partition from the global nodal force vector
        self.FGf = self.FG[0: self.NDOF];  print("FGf Partition:", self.FGf, sep="\n");
        print(self.FGf.shape);  _ = array_convert_to_unit_system(self.FGf, "imperial")
        self.PFf = self.structure_fef[0: self.NDOF]
        print("PFf Partition:", self.PFf, sep="\n");
        print(self.PFf.shape)
        from pyMAOS.quantity_utils import numpy_array_of_quantity_to_numpy_array_of_float64
        # Extract magnitudes for calculation
        Kff_magnitudes = numpy_array_of_quantity_to_numpy_array_of_float64(self.Kff)
        print("Kff_magnitudes:", Kff_magnitudes, sep="\n")
        # FGf_magnitudes = np.array([f.magnitude if hasattr(f, 'magnitude') else float(f) for f in self.FGf],
        #                           dtype=np.float64)
        # print("FGf_magnitudes:", FGf_magnitudes, sep="\n")
        # PFf_magnitudes = np.array([p.magnitude if hasattr(p, 'magnitude') else float(p) for p in self.PFf],
        #                           dtype=np.float64);
        # print(PFf_magnitudes)
        # print(f"DEBUG: Kff_magnitudes shape: {Kff_magnitudes.shape}")
        # print(f"DEBUG: FGf_magnitudes shape: {FGf_magnitudes.shape}")
        # print(f"DEBUG: PFf_magnitudes shape: {PFf_magnitudes.shape}")
        # Check if the Kff partition is singular
        if np.linalg.cond(Kff_magnitudes) > 1.0 / np.finfo(np.float64).eps:
            print("WARNING: Kff partition is singular or nearly singular. The system may not have a unique solution.")
            self._unstable = True
            return None
        # If we have units, we need to handle them carefully
        # Use scipy.linalg.solve to solve the linear system
        # This will return the displacement vector U in the same units as the right-hand side
        # If the right-hand side is a Pint Quantity, we need to extract the magnitudes
        # and reattach the units to the result.
        # If the right-hand side is a numpy array, we can use it directly.
        # Check if we're working with Pint Quantity objects
        import scipy.linalg as sla

        if hasattr(self.FGf[0], 'magnitude') or hasattr(self.PFf[0], 'magnitude') or True:

            # Store units from the right-hand side for later
            # Extract original units from the right-hand side
            rhs = self.FGf - self.PFf; print(rhs)
            _ = array_convert_to_unit_system(rhs, "imperial")



            # Convert to numpy array of Quantity objects
            from pyMAOS.quantity_utils import numpy_array_of_quantity_to_numpy_array_of_float64
            rhs_quantities = numpy_array_of_quantity_to_numpy_array_of_float64(rhs)
            print(f"DEBUG: Converted to array of Quantities with shape {rhs_quantities.shape}")

            # Solve using scipy.linalg
            U_magnitudes = sla.solve(Kff_magnitudes, rhs_quantities)
            from pyMAOS.quantity_utils import extract_units_from_quantities
            rhs_units = extract_units_from_quantities(rhs)
            print(f"DEBUG: Right-hand side units: {rhs_units}")
            # Example usage of the units array
            # for idx, unit in np.ndenumerate(rhs_units):
            #     if unit is not None:
            #         print(f"Element at {idx} has unit: {unit}")
            # Define and immediately apply the lambda function for each unit
            conjugate_units_container_list = [(lambda u: unit_manager.ureg.UnitsContainer(
                {'[length]': 2, '[mass]': 1, '[time]': -2}) / u.dimensionality)(u)
                               for u in rhs_units]
            # Create quantities with value 1.0 and the container units
            quantities = [unit_manager.ureg.Quantity(1.0, container) for container in conjugate_units_container_list]
            print(f"DEBUG: Created {len(quantities)} quantities from containers")

            # Convert to imperial system
            imperial_quantities = array_convert_to_unit_system(quantities, "imperial")
            print(f"DEBUG: Converted to imperial: {imperial_quantities}")

            # Extract just the units
            imperial_units = [q.units for q in imperial_quantities]
            print(f"DEBUG: Extracted units: {imperial_units}")

            print(f"DEBUG: Converted {rhs_units} to conjugate unit(s): {conjugate_units_container_list}")

            # Reattach conjugate units to displacement results
            self.U = np.array([pyMAOS.unit_manager.ureg.Quantity(mag, conj_unit)
                               for mag, conj_unit in zip(U_magnitudes, conjugate_units_container_list)], dtype=object)

            print(f"DEBUG: Reattached conjugate units to displacement vector: {self.U}")

        else:
            # No units involved, use scipy.linalg directly
            print("DEBUG: Using scipy.linalg directly (no units)")
            self.U = sla.solve(self.Kff, self.FGf - self.PFf)
        print(self.U)
        return self.U

    def set_node_displacements(self, load_combination=None):
        # Later, to restart from saved state:
        # if self.load_structure_state("structure_state.bin"):
        #     print("Successfully restored structure state - ready to continue analysis")
        # else:
        #     print("Failed to load state - need to recalculate")

        print("Structure displacementU:", self.U, sep="\n")#; print(self.U.shape)

        # Full Displacement Vector
        # Result is still mapped to DOF via FM
        # Create USTRUCT as an object array to hold Quantity objects
        # Create USTRUCT as self.U padded with zeros up to self.NJD * self.NJ elements
        USTRUCT = np.zeros(self.DIM, dtype=object)

        # Copy values from self.U to the beginning of padded_array
        USTRUCT[:len(self.U)] = self.U

        # Debug information
        print( f"DEBUG: Created USTRUCT by padding self.U ({len(self.U)} elements) to full size ({self.DIM} elements)")

        print("USTRUCT")
        _ = array_convert_to_unit_system(USTRUCT, "imperial")
        # store displacement results to the current case to the nodes
        for node in self.nodes:
            node_index = self.uid_to_index[node.uid]
            keys = self.FM[node_index * node.NJD: (node_index + 1) * self.NJD]
            node_displacements = USTRUCT[keys]

            print(f"node {node.uid}    Ux: {node_displacements[0]:.4E} -- Uy: {node_displacements[1]:.4E} -- Rz: {node_displacements[2]:.4E}")
            # print(f"DEBUG: Node {node.uid} displacements with units: {node_displacements}");

            node.displacements[load_combination.name] = node_displacements

        return


    def compute_reactions(self, load_combination):
        """Calculate nodal reactions for the given load combination"""
        for node in self.nodes:
            node_index = self.uid_to_index[node.uid]
            # Initialize reactions vector [rx, ry, mz]
            # reactions = np.zeros(self.NJD)
            #
            # # Add contributions from nodal loads (with negative sign)
            # for load_case, load in node.loads.items():
            #     load_factor = load_combination.factors.get(load_case, 0)
            #     reactions -= np.array(load) * load_factor
            #
            # # Add contributions from member end forces
            # for member in self.members:
            #     member_FG = member.set_end_forces_global(load_combination)
            #     print(f"Member {member.uid} Fixed End Forces:\n{member_FG}")
            #     # Add forces from i-node if this node is the i-node
            #     if member.inode == node:
            #         # Debug shapes when encountering errors
            #         reactions += member_FG[0:self.NJD]  # Use reshape(-1) to ensure correct shape
            #
            #     # Add forces from j-node if this node is the j-node
            #     if member.jnode == node:
            #         reactions += member_FG[self.NJD:(2 * self.NJD)]
            #
            # # Override with spring reactions if applicable
            # u = node.displacements.get(load_combination.name, np.zeros(self.NJD))
            # for i in range(self.NJD):
            #     if node._spring_stiffness[i] > 0:
            #         reactions[i] = -1 * u[i] * node._spring_stiffness[i]
            #
            # # Store reactions and print summary
            # node.reactions[load_combination.name] = reactions.tolist()
            node.compute_reactions(load_combination)


    def _verify_stable(self, FM, KSTRUCT):
        """
        Check the diagonal terms of the stiffness matrix against support
        conditions. If diagonal term is 0 and the node is unsupported for
        that DOF then the Kmatrix is singular and unstable.

        Returns
        -------
        If unstable returns a dictionary of unstable nodes
        and degree of freedom marked unstable.

        """

        unstablenodes = []

        for node_index, node in enumerate(self.nodes):
            # Check each DOF of node:
            for i, dof in enumerate(node.restraints):
                fmindex = node_index * self.NJD + i
                val = FM[fmindex]

                # value the diagonal position in the stiffness matrix
                kval = KSTRUCT[int(val), int(val)]

                if kval == 0 and dof != 1:
                    self._unstable = True
                    unstablenodes.append(
                        f"Node {node.uid} : Unstable for {node.restraints_key[i]}"
                    )
        # add unstable messages to ERROR list
        self._ERRORS.extend(unstablenodes)

        return unstablenodes


    def plot_loadcombos_vtk(self, loadcombos=None, scaling=None):
        """Visualizes the structure with results from multiple load combinations using VTK."""
        try:
            # Import the plotting function from R2Structure_extras
            from pyMAOS.structure2d_extras import plot_loadcombos_vtk, check_vtk_available

            # Check if VTK is available
            if not check_vtk_available():
                print("Warning: VTK library is not installed. Please install VTK for visualization.")
                return

            # Call the imported function with self as first argument
            plot_loadcombos_vtk(self, loadcombos, scaling)
        except ImportError as e:
            print(f"Warning: Visualization module not found: {e}")
            print("Make sure structure2d_extras.py is in the pyMAOS package directory.")
        except Exception as e:
            print(f"Error during visualization: {e}")


    def get_summary(self):
        """
        Returns a string representation of the structure using the display units
        specified in the input file.
        """
        # Access the global unit variables (defined elsewhere in the program)
        # import sys
        # from inspect import currentframe, getouterframes
        # module = sys.modules[__name__]

        # Attempt to get unit info from module scope
        # Use units from self.units dictionary with appropriate defaults
        force_unit = self.units.get('force', 'N')
        length_unit = self.units.get('length', 'm')
        pressure_unit = self.units.get('pressure', 'Pa')
        moment_unit = self.units.get('moment', f"{force_unit}*{length_unit}")
        distributed_load_unit = self.units.get('distributed_load', f"{force_unit}/{length_unit}")

        # Create a ureg for conversions if not already available
        try:
            from pyMAOS.pymaos_units import ureg
            Q_ = ureg.Quantity
        except:
            # Fall back if pint is not available
            print("Warning: pint library not available, using SI units for display")
            Q_ = lambda x, unit: x

        # Build the output string
        result = []
        result.append("=" * 80)
        result.append(f"STRUCTURAL MODEL SUMMARY")
        result.append(f"Display units: Force={force_unit}, Length={length_unit}, Pressure={pressure_unit}")
        result.append("-" * 80)

        # Basic structure info
        result.append(f"Number of nodes: {self.NJ}")
        result.append(f"Number of members: {self.NM}")
        result.append(f"Degrees of freedom: {self.NDOF}")
        result.append(f"Number of restraints: {self.NR}")

        # Node information
        result.append("\nNODE INFORMATION:")
        result.append(f"{'Node ID':8} {'X (' + length_unit + ')':15} {'Y (' + length_unit + ')':15} {'Restraints':15}")
        result.append("-" * 80)

        for node in sorted(self.nodes, key=operator.attrgetter('uid')):
            # Convert coordinates to display units
            x_display = Q_(node.x, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else node.x
            y_display = Q_(node.y, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else node.y

            # Format restraints
            restraint_str = "".join([
                "Rx" if node.restraints[0] else "--",
                " Ry" if node.restraints[1] else " --",
                " Rz" if node.restraints[2] else " --"
            ])

            result.append(f"{node.uid:<8} {x_display:<15.4g} {y_display:<15.4g} {restraint_str:<15}")

        # Member information
        result.append("\nMEMBER INFORMATION:")
        result.append(f"{'Member ID':10} {'Type':8} {'i-node':8} {'j-node':8} {'Length (' + length_unit + ')':15}")
        result.append("-" * 80)

        for member in sorted(self.members, key=lambda m: m.uid):  # type: ignore
            # Convert length to display units
            length_display = Q_(member.length, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else member.length

            result.append(
                f"{member.uid:<10} {member.type:<8} {member.inode.uid:<8} {member.jnode.uid:<8} {length_display:<15.4g}")

        # Material properties
        materials_seen = set()
        result.append("\nMATERIAL PROPERTIES:")
        result.append(f"{'Material':10} {'E (' + pressure_unit + ')':15}")
        result.append("-" * 80)

        for member in self.members:
            if member.material not in materials_seen:
                # Convert elastic modulus to display units
                E_display = Q_(member.material.E, 'Pa').to(pressure_unit).magnitude if hasattr(Q_,
                                                                                               'to') else member.material.E

                result.append(f"{member.material.uid:<10} {E_display:<15.4g}")
                materials_seen.add(member.material)

        # Section properties
        sections_seen = set()
        result.append("\nSECTION PROPERTIES:")
        result.append(f"{'Section':10} {'Area (' + length_unit + '^2)':18} {'Ixx (' + length_unit + '^4)':18}")
        result.append("-" * 80)

        for member in self.members:
            if member.section not in sections_seen:
                # Convert section properties to display units
                area_display = Q_(member.section.Area, 'm^2').to(length_unit + '^2').magnitude if hasattr(Q_,
                                                                                                          'to') else member.section.Area
                ixx_display = Q_(member.section.Ixx, 'm^4').to(length_unit + '^4').magnitude if hasattr(Q_,
                                                                                                        'to') else member.section.Ixx

                result.append(f"{member.section.uid:<10} {area_display:<18.4g} {ixx_display:<18.4g}")
                sections_seen.add(member.section)

        # Load information
        has_node_loads = any(node.loads for node in self.nodes)
        if has_node_loads:
            result.append("\nNODAL LOADS:")
            result.append(
                f"{'Node ID':8} {'Load Case':10} {'Fx (' + force_unit + ')':15} {'Fy (' + force_unit + ')':15} {'Mz (' + moment_unit + ')':15}")
            result.append("-" * 80)

            for node in sorted(self.nodes, key=lambda n: n.uid):  # type: ignore
                if node.loads:
                    for case, load in node.loads.items():
                        # Convert forces and moments to display units
                        fx_display = Q_(load[0], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[0]
                        fy_display = Q_(load[1], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[1]
                        mz_display = Q_(load[2], 'N*m').to(moment_unit).magnitude if hasattr(Q_, 'to') else load[2]

                        result.append(
                            f"{node.uid:<8} {case:<10} {fx_display:<15.4g} {fy_display:<15.4g} {mz_display:<15.4g}")

        # Displacements and reactions if calculated
        has_displacements = any(hasattr(node, 'displacements') and node.displacements for node in self.nodes)
        if has_displacements:
            result.append("\nNODAL DISPLACEMENTS (most recent load combination):")
            result.append(f"{'Node ID':8} {'Ux (' + length_unit + ')':15} {'Uy (' + length_unit + ')':15} {'Rz (rad)':15}")
            result.append("-" * 80)

            for node in sorted(self.nodes, key=lambda n: n.uid):  # type: ignore
                if hasattr(node, 'displacements') and node.displacements:
                    # Get the most recent load combination
                    latest_combo = list(node.displacements.keys())[-1]
                    disp = node.displacements[latest_combo]

                    # Convert displacements to display units
                    ux_display = Q_(disp[0], 'm').to(length_unit).magnitude
                    uy_display = Q_(disp[1], 'm').to(length_unit).magnitude
                    rz_display = Q_(disp[2], 'rad').to('rad').magnitude  # Radians are dimensionless in terms of conversion

                    result.append(f"{node.uid:<8} {ux_display:<15.6g} {uy_display:<15.6g} {rz_display:<15.6g}")

        # Errors if any
        if self._ERRORS:
            result.append("\nERRORS:")
            result.append("-" * 80)
            for error in self._ERRORS:
                result.append(error)

        # THIS IS THE KEY FIX - Return the joined result
        return "\n".join(result)


    def export_results_to_excel(self, output_file, loadcombos=None, **kwargs):
        """
        Export structural analysis results to Excel format with multiple sheets including visualization

        Parameters
        ----------
        output_file : str or Path
            Path for the output Excel file
        loadcombos : list of LoadCombo objects, optional
            List of load combinations to include in the export (if None, uses all analyzed load combinations)
        **kwargs : dict
            Additional options:
            - include_visualization : bool, default True
                Whether to include structure visualization sheet
            - unit_system : str, default None
                The unit system to use ("imperial", "si", "metric_kn")
                If None, uses the current unit system from the model
            - scaling : dict, optional
                Scaling factors for visualization
        """
        # Check for required packages
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from pathlib import Path
            from pint import UnitRegistry
        except ImportError as e:
            raise ImportError(f"Required package not available for Excel export: {e}")

        # Create unit registry for conversions
        
        Q_ = unit_manager.ureg.Quantity

        # Process unit system
        unit_system = kwargs.get('unit_system')
        if unit_system:
            # Import unit systems
            from pyMAOS.pymaos_units import SI_UNITS, IMPERIAL_DISPLAY_UNITS, METRIC_KN_UNITS, set_unit_system

            # Use the specified unit system for display
            if unit_system == "imperial":
                display_units = IMPERIAL_DISPLAY_UNITS
                system_name = "Imperial"
            elif unit_system == "si":
                display_units = SI_UNITS
                system_name = "SI"
            elif unit_system == "metric_kn":
                display_units = METRIC_KN_UNITS
                system_name = "Metric kN"
            else:
                display_units = self.units
                system_name = "Current"
        else:
            display_units = self.units
            system_name = "Current"

        print(f"Using {system_name} units for Excel export")

        # Utility function for unit conversion
        def convert_value(value, from_unit, to_unit):
            """Convert a value from one unit to another"""
            try:
                # Handle special case for dimensionless units like radians
                if to_unit in ['rad', 'radian', 'radians']:
                    return value

                # Convert using pint
                return Q_(value, from_unit).to(to_unit).magnitude
            except Exception as e:
                print(f"Warning: Could not convert {value} from {from_unit} to {to_unit}: {e}")
                return value

        # Resolve output file path
        output_file = Path(output_file)

        # Get list of all load combinations that have been analyzed
        if loadcombos is None:
            # Find all unique load combos that have results
            all_combos = set()
            for node in self.nodes:
                if hasattr(node, 'displacements'):
                    all_combos.update(node.displacements.keys())
            from pyMAOS.loadcombos import LoadCombo
            loadcombos = [LoadCombo(name, {name: 1.0}, [name], False, "CUSTOM") for name in all_combos]

        if not loadcombos:
            raise ValueError("No load combinations specified and no analysis results found.")

        # Extract options
        include_visualization = kwargs.get('include_visualization', True)

        print(f"Exporting analysis results to {output_file}...")

        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Create formats
            try:
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
            except AttributeError:
                header_format = None

            # 1. Summary sheet
            summary_data = {
                'Parameter': ['Number of Nodes', 'Number of Members', 'Degrees of Freedom',
                              'Number of Restraints', 'Analysis Type'],
                'Value': [self.NJ, self.NM, self.NDOF, self.NR, 'Linear Static']
            }
            # Add load combinations to summary
            for i, combo in enumerate(loadcombos):
                summary_data['Parameter'].append(f"Load Combination {i + 1}")
                summary_data['Value'].append(combo.name)

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # 2. Structure visualization (if requested)
            if include_visualization:
                try:
                    from pyMAOS.structure2d_plot_with_matplotlib import plot_structure_matplotlib

                    # Use the existing plot function
                    fig, ax = plot_structure_matplotlib(self.nodes, self.members)

                    # Add visualization to Excel
                    worksheet = workbook.add_worksheet('Structure Visualization')

                    # Save the figure to a BytesIO object
                    imgdata = io.BytesIO()
                    fig.savefig(imgdata, format='png', dpi=150, bbox_inches='tight')
                    imgdata.seek(0)

                    # Insert the image into the worksheet
                    worksheet.insert_image('A1', 'structure.png',
                                           {'image_data': imgdata, 'x_scale': 0.8, 'y_scale': 0.8})

                    # Close the matplotlib figure to free memory
                    plt.close(fig)

                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not create structure visualization: {e}")

            # 3. Units sheet
            units_data = []
            for dimension, unit in display_units.items():
                units_data.append({'Dimension': dimension, 'Unit': unit})
            units_df = pd.DataFrame(units_data)
            units_df.to_excel(writer, sheet_name='Units', index=False)

            # Process each load combination
            for combo in loadcombos:
                combo_name = combo.name
                sheet_name = f"Results_{combo_name}"[:31]  # Excel sheet name limit is 31 chars

                # 4. Node information sheet for this load combination
                nodes_data = []
                for node in sorted(self.nodes, key=lambda n: n.uid):
                    # Convert coordinates to display units
                    x_display = convert_value(node.x, 'm', display_units['length'])
                    y_display = convert_value(node.y, 'm', display_units['length'])

                    node_info = {
                        'Node ID': node.uid,
                        f'X ({display_units["length"]})': x_display,
                        f'Y ({display_units["length"]})': y_display,
                        'Restrained X': node.restraints[0],
                        'Restrained Y': node.restraints[1],
                        'Restrained Z': node.restraints[2],
                    }

                    # Add displacements if available
                    if hasattr(node, 'displacements') and combo_name in node.displacements:
                        disp = node.displacements[combo_name]
                        # Convert to display units
                        disp_x = convert_value(disp[0], 'm', display_units['length'])
                        disp_y = convert_value(disp[1], 'm', display_units['length'])
                        rot_z = disp[2]  # Radians are dimensionless

                        node_info.update({
                            f'Displacement X ({display_units["length"]})': disp_x,
                            f'Displacement Y ({display_units["length"]})': disp_y,
                            'Rotation Z (rad)': rot_z,
                        })
                    else:
                        node_info.update({
                            f'Displacement X ({display_units["length"]})': 0.0,
                            f'Displacement Y ({display_units["length"]})': 0.0,
                            'Rotation Z (rad)': 0.0,
                        })

                    # Add reactions if available
                    if hasattr(node, 'reactions') and combo_name in node.reactions:
                        reaction = node.reactions[combo_name]
                        # Convert to display units
                        rx = convert_value(reaction[0], 'N', display_units['force'])
                        ry = convert_value(reaction[1], 'N', display_units['force'])
                        mz = convert_value(reaction[2], 'N*m', display_units['moment'])

                        node_info.update({
                            f'Reaction X ({display_units["force"]})': rx,
                            f'Reaction Y ({display_units["force"]})': ry,
                            f'Moment Z ({display_units["moment"]})': mz,
                        })
                    else:
                        node_info.update({
                            f'Reaction X ({display_units["force"]})': 0.0,
                            f'Reaction Y ({display_units["force"]})': 0.0,
                            f'Moment Z ({display_units["moment"]})': 0.0,
                        })

                    nodes_data.append(node_info)

                nodes_df = pd.DataFrame(nodes_data)
                nodes_df.to_excel(writer, sheet_name=sheet_name, index=False)

                # 5. Member forces sheet for this load combination
                forces_data = []
                for member in sorted(self.members, key=lambda m: m.uid):
                    # Calculate member forces for this load combination
                    if hasattr(member, 'end_forces_global') and combo_name in member.end_forces_global:
                        global_forces = member.end_forces_global[combo_name]
                        # Ensure global_forces is 1D
                        global_forces = np.asarray(global_forces).flatten()
                    else:
                        # Calculate forces if not already available
                        try:
                            global_forces = member.set_end_forces_global(combo)
                            # Ensure global_forces is 1D
                            global_forces = np.asarray(global_forces).flatten()
                        except:
                            global_forces = np.zeros(6)

                    if hasattr(member, 'end_forces_local') and combo_name in member.end_forces_local:
                        # Always flatten local_forces to ensure consistent 1D access
                        local_forces = np.asarray(member.end_forces_local[combo_name]).flatten()
                    else:
                        try:
                            member.Flocal(combo)
                            # Always flatten the result to get a 1D array
                            local_forces = np.asarray(member.end_forces_local[combo_name]).flatten()
                        except:
                            local_forces = np.zeros(6)

                    # Convert forces to display units
                    global_forces_display = [
                        convert_value(global_forces[0], 'N', display_units['force']),
                        convert_value(global_forces[1], 'N', display_units['force']),
                        convert_value(global_forces[2], 'N*m', display_units['moment']),
                        convert_value(global_forces[3], 'N', display_units['force']),
                        convert_value(global_forces[4], 'N', display_units['force']),
                        convert_value(global_forces[5], 'N*m', display_units['moment'])
                    ]

                    local_forces_display = [
                        convert_value(local_forces[0], 'N', display_units['force']),
                        convert_value(local_forces[1], 'N', display_units['force']),
                        convert_value(local_forces[2], 'N*m', display_units['moment']),
                        convert_value(local_forces[3], 'N', display_units['force']),
                        convert_value(local_forces[4], 'N', display_units['force']),
                        convert_value(local_forces[5], 'N*m', display_units['moment'])
                    ]

                    # i-node global forces
                    forces_data.append({
                        'Member ID': member.uid,
                        'Node': f"{member.inode.uid} (i)",
                        f'Fx ({display_units["force"]})': global_forces_display[0],
                        f'Fy ({display_units["force"]})': global_forces_display[1],
                        f'Mz ({display_units["moment"]})': global_forces_display[2],
                        'System': 'Global'
                    })

                    # j-node global forces
                    forces_data.append({
                        'Member ID': member.uid,
                        'Node': f"{member.jnode.uid} (j)",
                        f'Fx ({display_units["force"]})': global_forces_display[3],
                        f'Fy ({display_units["force"]})': global_forces_display[4],
                        f'Mz ({display_units["moment"]})': global_forces_display[5],
                        'System': 'Global'
                    })

                    # i-node local forces
                    forces_data.append({
                        'Member ID': member.uid,
                        'Node': f"{member.inode.uid} (i)",
                        f'Fx ({display_units["force"]})': local_forces_display[0],
                        f'Fy ({display_units["force"]})': local_forces_display[1],
                        f'Mz ({display_units["moment"]})': local_forces_display[2],
                        'System': 'Local'
                    })

                    # j-node local forces
                    forces_data.append({
                        'Member ID': member.uid,
                        'Node': f"{member.jnode.uid} (j)",
                        f'Fx ({display_units["force"]})': local_forces_display[3],
                        f'Fy ({display_units["force"]})': local_forces_display[4],
                        f'Mz ({display_units["moment"]})': local_forces_display[5],
                        'System': 'Local'
                    })

                # Write member forces to a separate sheet for this load combo
                forces_df = pd.DataFrame(forces_data)
                sheet_name = f"Forces_{combo_name}"[:31]  # Excel sheet name limit is 31 chars
                forces_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Add a member properties sheet (common to all load combos)
            members_data = []
            for member in sorted(self.members, key=lambda m: m.uid):
                # Convert member properties to display units
                length_display = convert_value(member.length, 'm', display_units['length'])
                e_display = convert_value(member.material.E, 'Pa', display_units['pressure'])
                area_display = convert_value(member.section.Area, 'm^2', f"{display_units['length']}^2")
                ixx_display = convert_value(member.section.Ixx, 'm^4', f"{display_units['length']}^4")

                member_info = {
                    'Member ID': member.uid,
                    'Type': member.type,
                    'i-node': member.inode.uid,
                    'j-node': member.jnode.uid,
                    f'Length ({display_units["length"]})': length_display,
                    'Material ID': member.material.uid,
                    f'E ({display_units["pressure"]})': e_display,
                    'Section ID': member.section.uid,
                    f'Area ({display_units["length"]}²)': area_display,
                    f'Ixx ({display_units["length"]}⁴)': ixx_display,
                }

                # Add hinge information if it's a frame
                if hasattr(member, 'hinges'):
                    hinge_info = []
                    if member.hinges[0]:
                        hinge_info.append('i-node')
                    if member.hinges[1]:
                        hinge_info.append('j-node')
                    member_info['Hinges'] = ', '.join(hinge_info) if hinge_info else 'None'

                members_data.append(member_info)

            members_df = pd.DataFrame(members_data)
            members_df.to_excel(writer, sheet_name='Member Properties', index=False)

        print(f"Successfully exported results to {output_file}")
        return str(output_file)

    def verify_structure_state(self, state_data):
        """
        Verify that the loaded structure state is complete and consistent.
        Checks if K·U = F-P equation is satisfied.

        Parameters
        ----------
        state_data : dict
            Dictionary containing the loaded structure state

        Returns
        -------
        bool
            True if verification passes, False otherwise
        """
        print("Verifying structure state completeness and consistency...")

        # 1. Check if all essential components are present
        essential_keys = ['nodes', 'members', 'U', 'KSTRUCT', 'FG', 'structure_fef']
        missing_keys = [key for key in essential_keys if key not in state_data]
        if missing_keys:
            print(f"ERROR: Missing essential data: {missing_keys}")
            return False

        # 2. Check stiffness equation K·U = F-P for free DOFs
        if all(hasattr(self, attr) for attr in ['Kff', 'U', 'FGf', 'PFf']):
            try:
                # Convert quantities to numerical values for calculation
                from pyMAOS.quantity_utils import numpy_array_of_quantity_to_numpy_array_of_float64
                import numpy as np

                # Extract magnitudes for calculation
                Kff_mag = numpy_array_of_quantity_to_numpy_array_of_float64(self.Kff)
                U_mag = numpy_array_of_quantity_to_numpy_array_of_float64(self.U)
                FGf_mag = numpy_array_of_quantity_to_numpy_array_of_float64(self.FGf)
                PFf_mag = numpy_array_of_quantity_to_numpy_array_of_float64(self.PFf)

                # Calculate left and right sides of equation
                KU = np.matmul(Kff_mag, U_mag)
                F_minus_P = FGf_mag - PFf_mag

                # Calculate residual and relative error
                residual = KU - F_minus_P
                error_norm = np.linalg.norm(residual)
                rel_error = error_norm / np.linalg.norm(F_minus_P) if np.linalg.norm(F_minus_P) > 1e-10 else error_norm

                print(f"Stiffness equation verification:")
                print(f"  Norm of K·U:   {np.linalg.norm(KU):.6e}")
                print(f"  Norm of F-P:   {np.linalg.norm(F_minus_P):.6e}")
                print(f"  Residual norm: {error_norm:.6e}")
                print(f"  Relative error: {rel_error:.6e}")

                # Check if error is acceptably small (adjust tolerance as needed)
                if rel_error > 1e-6:
                    print("WARNING: Relative error in stiffness equation exceeds tolerance")
                    return False

                print("✓ Stiffness equation verified successfully")

            except Exception as e:
                print(f"ERROR during verification: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("WARNING: Cannot verify stiffness equation - missing required data")

        # 3. Check node equilibrium by verifying reactions match applied loads + member forces
        # (This would require gathering all member end forces for each node)

        print("Structure state verification completed")
        return True

    def save_structure_state(self, filename):
        """
        Save the entire structure object to a binary file in one operation.

        Parameters
        ----------
        filename : str
            Path to the binary output file

        Returns
        -------
        bool
            True if saving was successful, False otherwise
        """
        import pickle
        import zlib

        print(f"Saving complete structure state to {filename}...")

        try:
            # Pickle the entire object
            pickled_data = pickle.dumps(self)

            # Compress the pickled data
            compressed_data = zlib.compress(pickled_data)

            # Write compressed data to file
            with open(filename, 'wb') as file:
                file.write(compressed_data)

            print(f"Structure state saved successfully ({len(compressed_data):,} bytes compressed, {len(pickled_data):,} bytes uncompressed)")
            return True

        except Exception as e:
            print(f"Error saving structure state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_structure_state(self, filename):
        """
        Load structure state from a binary file

        Parameters
        ----------
        filename : str
            Path to the binary file

        Returns
        -------
        bool
            True if loading was successful, False otherwise
        """
        import pickle
        import zlib
        import io
        import pint
        import sys

        print(f"Loading structure state from {filename}...")

        try:
            # Read the compressed data
            with open(filename, 'rb') as file:
                compressed_data = file.read()

            # Define a custom unpickler for Pint Quantity objects
            class QuantityUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle Pint Quantity objects specially
                    if module == 'pint.quantity' and name == 'Quantity':
                        return pyMAOS.unit_manager.ureg.Quantity
                    return super().find_class(module, name)

            try:
                # Try to decompress the data
                pickled_data = zlib.decompress(compressed_data)
                print("Data is compressed, decompressing...")

                # Use custom unpickler
                unpickler = QuantityUnpickler(io.BytesIO(pickled_data))
                loaded_structure = unpickler.load()
                print("Successfully loaded compressed state data")

            except zlib.error:
                # If decompression fails, try as raw pickle
                print("Data doesn't appear to be compressed, trying direct pickle")
                unpickler = QuantityUnpickler(io.BytesIO(compressed_data))
                loaded_structure = unpickler.load()

            # Transfer all attributes from the loaded structure to self
            for attr_name, attr_value in vars(loaded_structure).items():
                setattr(self, attr_name, attr_value)

            # Verify the state is consistent
            if hasattr(self, 'verify_structure_state'):
                if self.verify_structure_state(vars(self)):
                    print("Structure state verified successfully")
                else:
                    print("Warning: Structure state verification failed")

            # Rebuild node and member mappings
            self.create_uid_maps()

            print("Structure state loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading structure state: {e}")
            import traceback
            traceback.print_exc()
            return False