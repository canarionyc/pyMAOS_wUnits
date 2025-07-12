# -*- coding: utf-8 -*-
import os
import numpy as np


class R2Structure:
    def __init__(self, nodes, members):
        self.nodes = nodes
        self.create_uid_map()  # Create UID to index mapping
        self.members = members

        # Structure Type
        # 2D Structure
        # Ux,Uy, and Rz = 3
        # Number of possible Joint Displacements
        self.NJD = 3

        # Number of Joints
        self.NJ = len(self.nodes)

        # Number of Members
        self.NM = len(self.members)

        # Number of Restraints
        self.NR = sum([sum(node.restraints) for node in self.nodes])

        # Degrees of Freedom
        self.NDOF = (self.NJD * self.NJ) - self.NR

        # Data Stores
        self._springNodes = None
        self._nonlinearNodes = None
        self._D = {}  # Structure Displacement Vector Dictionary

        # Flags
        self._unstable = False
        self._Kgenerated = False
        self._ERRORS = []

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
        self.uid_to_index = {node.uid: node_index for node_index, node in enumerate(self.nodes)}

    def freedom_map(self):
        # Freedom Map
        FM = np.zeros(self.NJD * self.NJ, dtype=np.int32)  # Ensure FM is initialized as an integer array

        # Loop through the nodes mapping free and restrained joint displacements to
        # the Freedom Map (FM). This will facilitate generating the global stiffness
        # matrix in partitioned form.

        j = 0  # starting index for the first free displacement
        k = self.NDOF  # starting index for the first restraint

        for node_index, node in enumerate(self.nodes):
            for r, restraint in enumerate(node.restraints):
                fmindex = node_index * self.NJD + r

                if restraint == 0:
                    FM[fmindex] = j
                    j += 1
                else:
                    FM[fmindex] = k
                    k += 1

        return FM.astype(np.int32)  # Ensure FM is returned as an integer array

    
    def Kstructure(self, FM, **kwargs):
        """
        Build the structure stiffness matrix organized into paritioned form
        using the freedom map to reposition nodal DOFs

        Returns
        -------
        KSTRUCT: Numpy Matrix
            Structure Stiffness Matrix.

        """

        output_dir=kwargs.get('output_dir', '.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Structure Stiffness Matrix
        KSTRUCT = np.zeros([self.NJD * self.NJ, self.NJD * self.NJ])


        for member in self.members:
            # Member global stiffness matrix
            kmglobal = member.kglobal()
            print(f"Member {member.uid}:\n"); print(kmglobal, flush=True)
            # Check if the member is a truss
                
            # Save each member's global stiffness matrix to a separate CSV file
            filename = os.path.join(output_dir, f'member_{member.uid}_global_stiffness.csv')
            np.savetxt(filename, kmglobal, delimiter=',', fmt='%g')
            print(f"Saved member {member.uid} global stiffness matrix to {filename}", flush=True)


            # Freedom map for i and j nodes
            imap = [ int(FM[(self.uid_to_index[member.inode.uid]) * self.NJD + r]) for r in range(self.NJD) ]
            imap.extend([int(FM[(self.uid_to_index[member.jnode.uid]) * self.NJD + r]) for r in range(self.NJD)])
            print(f"Freedom Map Indices for Member {member.uid}:\n{imap}", flush=True)
            for i in range(self.NJD):
                for j in range(self.NJD): 
                    print(f"i: {i}, j: {j}, imap[i]: {imap[i]}, imap[j]: {imap[j]}", flush=True)
                    KSTRUCT[imap[i], imap[j]] += kmglobal[i, j]
                    KSTRUCT[imap[i + self.NJD], imap[j]] += kmglobal[i + self.NJD, j]
                    KSTRUCT[imap[i], imap[j + self.NJD]] += kmglobal[i, j + self.NJD]
                    KSTRUCT[imap[i + self.NJD], imap[j + self.NJD]] += kmglobal[i + self.NJD, j + self.NJD]

        # Loop through Spring Nodes and add the spring stiffness
        if self._springNodes:
            for node in self._springNodes:
                uxposition = int(FM[(node.uid - 1) * self.NJD + 0])
                uyposition = int(FM[(node.uid - 1) * self.NJD + 1])
                rzposition = int(FM[(node.uid - 1) * self.NJD + 2])
                kux = node._spring_stiffness[0]
                kuy = node._spring_stiffness[1]
                krz = node._spring_stiffness[2]

                KSTRUCT[uxposition, uxposition] += kux
                KSTRUCT[uyposition, uyposition] += kuy
                KSTRUCT[rzposition, rzposition] += krz
        self._Kgenerated = True
        return KSTRUCT

    def nodal_force_vector(self, FM, load_combination):
        """
        Build the structure nodal force vector mapped to the same partitions
        as KSTRUCT using the freedom map (FM).

        Returns
        -------
        FG : Numpy Array
            Structure Nodal Force Vector.

        """
        FG = np.zeros(self.NJD * self.NJ)

        for node_index,node in enumerate(self.nodes):
            for load_case, load in node.loads.items():
                load_factor = load_combination.factors.get(load_case, 0)
                factored_load = [load_factor * l for l in load]

                for i, f in enumerate(factored_load):
                    fmindex = node_index * self.NJD + i

                    FG[int(FM[fmindex])] += f
        return FG

    def member_fixed_end_force_vector(self, FM, load_combination):
        PF = np.zeros(self.NJD * self.NJ)

        for member in self.members:
            if member.type != "TRUSS":
                Ff = member.FEFglobal(load_combination)
                i_index = self.uid_to_index[member.inode.uid]
                j_index = self.uid_to_index[member.jnode.uid]
                fmindexi = i_index * self.NJD
                fmindexj = j_index * self.NJD

                PF[int(FM[fmindexi])] += Ff[0, 0]
                PF[int(FM[fmindexi + 1])] += Ff[0, 1]
                PF[int(FM[fmindexi + 2])] += Ff[0, 2]
                PF[int(FM[fmindexj])] += Ff[0, 3]
                PF[int(FM[fmindexj + 1])] += Ff[0, 4]
                PF[int(FM[fmindexj + 2])] += Ff[0, 5]

        return PF

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
        debug = kwargs.get('debug', False)
        if debug:
            print("--- Running in Debug Mode ---")

        # Generate Freedom Map
        FM = self.freedom_map()
        print("Freedom Map:\n",FM)

        # Generate Full Structure Stiffness Matrix
        KSTRUCT = self.Kstructure(FM)

        if debug:
            print("K_new:\n", KSTRUCT); print(KSTRUCT.shape)
            output_dir = kwargs.get('output_dir', '.')
            np.savetxt(os.path.join(output_dir, 'K_new.csv'), KSTRUCT, delimiter=',', fmt='%lg')

        self._verify_stable(FM, KSTRUCT)

        if self._unstable:
            return 0
        else:
            # Build Nodal Force Vector
            FG = self.nodal_force_vector(FM, load_combination)
            print("Nodal Force Vector:\n", FG)
            # Build Member Fixed-end-Force vector
            PF = self.member_fixed_end_force_vector(FM, load_combination)
            print("Member Fixed End Force Vector:\n", PF); print(PF.shape)
            # Slice out the Kff partition from the global structure stiffness
            # Matrix
            self.Kff = KSTRUCT[0 : self.NDOF, 0 : self.NDOF]
            print("Kff Partition:\n", self.Kff); print(self.Kff.shape)
            # Slice out the FGf partition from the global nodal force vector
            self.FGf = FG[0 : self.NDOF]
            print("FGf Partition:\n", self.FGf); print(self.FGf.shape)
            self.PFf = PF[0 : self.NDOF]
            print("PFf Partition:\n", self.PFf); print(self.PFf.shape)

            # Use Numpy linear Algebra solve function to solve for the
            # displacements at the free nodes.
            U = np.linalg.solve(self.Kff, self.FGf - self.PFf)
            print("Displacement Vector U:\n", U); print(U.shape)
            # Full Displacement Vector
            # Result is still mapped to DOF via FM
            USTRUCT = np.zeros(self.NJD * self.NJ)

            # Add the resulting free displacements to the appropriate spots in
            # the Full displacement vector
            USTRUCT += np.pad(U, (0, self.NJD * self.NJ - np.shape(U)[0]))
            print("Full Displacement Vector USTRUCT:\n", USTRUCT); print(USTRUCT.shape) 
            # store displacement results to the current case to the nodes
            for node_index,node in enumerate(self.nodes):
                print(f"Node {node.uid} Displacements:")
                uxindex = int(FM[node_index * self.NJD + 0])
                uyindex = int(FM[node_index * self.NJD + 1])
                rzindex = int(FM[node_index * self.NJD + 2])

                node_displacements = [
                    USTRUCT[uxindex],
                    USTRUCT[uyindex],
                    USTRUCT[rzindex],
                ]

                node.displacements[load_combination.name] = node_displacements
                print(f"    Ux: {node_displacements[0]:.4E} -- Uy: {node_displacements[1]:.4E} -- Rz: {node_displacements[2]:.4E}")
            # compute reactions
            self.compute_reactions(load_combination)

            return U

    def compute_reactions(self, load_combination):
        # Compute Reactions
        for node in self.nodes:
            print(f"Computing reactions for Node {node.uid}")
            rx = 0
            ry = 0
            mz = 0

            for load_case, load in node.loads.items():
                print(f"    Load Case: {load_case} Load: {load}")
                load_factor = load_combination.factors.get(load_case, 0)
                FNL = [load_factor * i for i in load]

                rx += -1 * FNL[0]
                ry += -1 * FNL[1]
                mz += -1 * FNL[2]

            for member in self.members:
                print(f"    Member {member.uid} Global Forces:")
                # Get the member global forces for the current load combination
                member_FG = member.Fglobal(load_combination)
                print(f"    Member {member.uid} Forces:\n{member_FG}")
                if member.inode == node:
                    rx += member_FG[0, 0]
                    ry += member_FG[0, 1]
                    mz += member_FG[0, 2]
                if member.jnode == node:
                    rx += member_FG[0, 3]
                    ry += member_FG[0, 4]
                    mz += member_FG[0, 5]

            # overide spring reactions to be k*displacement
            # Rework this when adding in uni-directional springs
            u = node.displacements.get(load_combination.name, [0, 0, 0])

            if node._spring_stiffness[0] > 0:
                rx = -1 * u[0] * node._spring_stiffness[0]

            if node._spring_stiffness[1] > 0:
                ry = -1 * u[1] * node._spring_stiffness[1]

            if node._spring_stiffness[2] > 0:
                mz = -1 * u[2] * node._spring_stiffness[2]

            node.reactions[load_combination.name] = [rx, ry, mz]
            print(f"    Reactions: Rx: {rx:.4E} -- Ry: {ry:.4E} -- Mz: {mz:.4E}")

    def _verify_stable(self, FM, KSTRUCT):
        """
        Check the diagonal terms of the stiffness matrix against support
        conditions If diagonal term is 0 and the node is unsupported for
        that DOF then the Kmatrix is singular and unstable.

        Returns
        -------
        If unstable returns a dictionary of unstable nodes
        and degree of freedom marked unstable.

        """

        unstablenodes = []

        for node_index,node in enumerate(self.nodes):
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
