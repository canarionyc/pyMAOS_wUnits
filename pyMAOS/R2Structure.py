# -*- coding: utf-8 -*-
import os
# from xml.dom import NO_DATA_ALLOWED_ERR
import numpy as np
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal')

import ast

# Add custom formatters for different numeric types
def format_with_dots(x): return '.'.center(9) if abs(x) < 1e-10 else f"{x:9.4g}"
def format_double(x): return '.'.center(13) if abs(x) < 1e-10 else f"{x:13.8lg}"  # More precision for doubles

np.set_printoptions( precision=4, 
    suppress=False, 
    formatter={
    'float': format_with_dots,     # For float32 and generic floats
    'float_kind': format_with_dots,  # For all floating point types
    'float64': format_double       # Specifically for float64 (double precision)
},
    linewidth=120  # Wider output to prevent unnecessary wrapping
)

# Parameters you can adjust:
# •	precision: Number of decimal places (4 is a good default)
# •	suppress: When True, very small values near zero are displayed as 0
# •	floatmode: Controls the display format
# •	'maxprec_equal': Best option for automatic switching between formats
# •	'fixed': Always use fixed-point notation
# •	'scientific': Always use scientific notation
# •	'unique': Use minimum digits to distinguish values

class R2Structure:
    def __init__(self, nodes, members):
        self.nodes = nodes
        self.create_uid_maps()  # Replace create_uid_map() with the new method
        
        # Validate that all member nodes exist in the node list
        for member in members:
            if member.inode.uid not in self.uid_to_index:
                raise ValueError(f"Member {member.uid} references node {member.inode.uid} which doesn't exist in the structure")
            if member.jnode.uid not in self.uid_to_index:
                raise ValueError(f"Member {member.uid} references node {member.jnode.uid} which doesn't exist in the structure")
        
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
        self.uid_to_index = {node.uid: i for i, node in enumerate(self.nodes)}

    def create_uid_maps(self):
        """Create mappings between UIDs and positions (in both directions)"""
        # Map from UID to position index
        self.uid_to_index = {node.uid: i for i, node in enumerate(self.nodes)}
        
        # Map from position index to UID
        self.index_to_uid = {i: node.uid for i, node in enumerate(self.nodes)}

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
        print(KSTRUCT.shape, flush=True)

        for member in self.members:
            # Member global stiffness matrix
            kmglobal = member.kglobal()
            print(f"Member {member.uid}:\tkmglobal\n{kmglobal}\n", flush=True)
            # Check if the member is a truss
            if member.type == "TRUSS":
                # For truss elements, verify that all bending/shear terms are zero
                bending_terms = [
                    kmglobal[2, 2],  # i-node bending terms
                    kmglobal[5, 5]   # j-node bending terms
                ]
                
                # Check if any of these terms are non-zero
                if not all(abs(term) < 1e-10 for term in bending_terms):
                    non_zero_terms = [f"[{i}]={val}" for i, val in enumerate(bending_terms) if abs(val) > 1e-10]
                    error_msg = (f"Error: Member {member.uid} is a TRUSS but has non-zero bending terms: "
                                f"{', '.join(non_zero_terms)}. Truss elements must have zero bending stiffness.")
                    print(error_msg)
                    raise ValueError(error_msg)
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
                    # print(f"i: {i}, j: {j}, imap[i]: {imap[i]}, imap[j]: {imap[j]}", flush=True)
                    KSTRUCT[imap[i], imap[j]] += kmglobal[i, j]
                    KSTRUCT[imap[i + self.NJD], imap[j]] += kmglobal[i + self.NJD, j]
                    KSTRUCT[imap[i], imap[j + self.NJD]] += kmglobal[i, j + self.NJD]
                    KSTRUCT[imap[i + self.NJD], imap[j + self.NJD]] += kmglobal[i + self.NJD, j + self.NJD]

        # Loop through Spring Nodes and add the spring stiffness
        if self._springNodes:
            for node in self._springNodes:
                node_index= self.uid_to_index[node.uid]
                uxposition = int(FM[(node_index) * self.NJD + 0])
                uyposition = int(FM[(node_index) * self.NJD + 1])
                rzposition = int(FM[(node_index) * self.NJD + 2])
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
                # fmindexi = i_index * self.NJD
                # fmindexj = j_index * self.NJD

                # PF[int(FM[fmindexi])] += Ff[0]
                # PF[int(FM[fmindexi + 1])] += Ff[1]
                # PF[int(FM[fmindexi + 2])] += Ff[2]
                # PF[int(FM[fmindexj])] += Ff[3]
                # PF[int(FM[fmindexj + 1])] += Ff[4]
                # PF[int(FM[fmindexj + 2])] += Ff[5]

                PF[FM[i_index * self.NJD:(i_index+1)  * self.NJD]] += Ff[0:3]
                PF[FM[j_index * self.NJD:(j_index+1)  * self.NJD]] += Ff[3:6]
        return PF

#     How Member Loads Are Correctly Handled
# Member loads (like distributed loads or point loads on members) don't modify the stiffness matrix. Instead, they contribute to the force vector through fixed-end forces (FEF). This is correct behavior in the matrix stiffness method:
# 1.	Stiffness Matrix (K): Represents structural properties (geometry, material, connectivity)
# 2.	Force Vector (F): Contains external forces, including equivalent nodal forces from member loads
# In your code:
# •	Kstructure(FM) builds the stiffness matrix from member geometry/properties
# •	member_fixed_end_force_vector(FM, load_combination) processes member loads into the PF vector
# •	The equation solved is K*u = F-PF where PF is the fixed-end forces from member loads
# Member loads are properly handled in solve_linear_static():

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
        if verbose:
            print("--- Running in Verbose Mode ---")

        # Generate Freedom Map
        FM = self.freedom_map()
        print("Freedom Map:\n",FM)

        # Generate Full Structure Stiffness Matrix
        KSTRUCT = self.Kstructure(FM)

        if verbose:
            print("K:\n", KSTRUCT); print(KSTRUCT.shape)
            output_dir = kwargs.get('output_dir', '.');  KSTRUCT_csv=os.path.join(output_dir, 'KSTRUCT.csv')
            np.savetxt(KSTRUCT_csv, KSTRUCT, delimiter=',', fmt='%lg')
            print(f"Saved KSTRUCT to {KSTRUCT_csv}")
        self._verify_stable(FM, KSTRUCT)

        if self._unstable:
            raise ValueError("Structure is unstable")
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
            # print("Full Displacement Vector USTRUCT:\n", USTRUCT); print(USTRUCT.shape) 
            # store displacement results to the current case to the nodes
            for node_index,node in enumerate(self.nodes):
                node_displacements=USTRUCT[FM[node_index * self.NJD : (node_index + 1) * self.NJD]].tolist()
                print(f"    Ux: {node_displacements[0]:.4E} -- Uy: {node_displacements[1]:.4E} -- Rz: {node_displacements[2]:.4E}")
                node.displacements[load_combination.name] = node_displacements
                
            # compute reactions
            self.compute_reactions(load_combination)

            return U

    def compute_reactions(self, load_combination):
        """Calculate nodal reactions for the given load combination"""
        for node in self.nodes:
            # Initialize reactions vector [rx, ry, mz]
            reactions = np.zeros(self.NJD)
            
            # Add contributions from nodal loads (with negative sign)
            for load_case, load in node.loads.items():
                load_factor = load_combination.factors.get(load_case, 0)
                reactions -= np.array(load) * load_factor
            
            # Add contributions from member end forces
            for member in self.members:
                member_FG = member.Fglobal(load_combination)
                print(f"Member {member.uid} Fixed End Forces:\n{member_FG}")
                # Add forces from i-node if this node is the i-node
                if member.inode == node:
                    # Debug shapes when encountering errors
                    reactions += member_FG[0:self.NJD]  # Use reshape(-1) to ensure correct shape
                    
                # Add forces from j-node if this node is the j-node
                if member.jnode == node:
                    reactions += member_FG[self.NJD:(2*self.NJD)]  # Use flatten() to ensure correct shape

            # Override with spring reactions if applicable
            u = node.displacements.get(load_combination.name, np.zeros(self.NJD))
            for i in range(self.NJD):
                if node._spring_stiffness[i] > 0:
                    reactions[i] = -1 * u[i] * node._spring_stiffness[i]
                    
            # Store reactions and print summary
            node.reactions[load_combination.name] = reactions.tolist()
            print(f"Node {node.uid} reactions: Rx={reactions[0]:.4E}, Ry={reactions[1]:.4E}, Mz={reactions[2]:.4E}")

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

    def plot_loadcombos_vtk(self, loadcombos=None, scaling=None):
        """
        Visualizes the structure with results from multiple load combinations using VTK.
    
        Parameters
        ----------
        loadcombos : list or single LoadCombo, optional
            The load combinations to display.
        scaling : dict, optional
            Scaling factors for deformations.
        """
        import vtk
        import math
    
        # Convert single loadcombo to list for consistent handling
        if loadcombos and not isinstance(loadcombos, list):
            loadcombos = [loadcombos]
        
        # --- Create base geometry (same for all load combinations) ---
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        line_colors = vtk.vtkUnsignedCharArray()
        line_colors.SetNumberOfComponents(3)
        line_colors.SetName("Colors")
    
        type_color_map = {"FRAME": (0, 0, 255), "TRUSS": (0, 255, 0)}
        default_color = (0, 0, 0)
    
        # Use the class's nodes and members
        node_uid_to_vtk_id = {node.uid: i for i, node in enumerate(self.nodes)}
        for node in self.nodes:
            points.InsertNextPoint(node.x, node.y, 0)
    
        for member in self.members:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, node_uid_to_vtk_id[member.inode.uid])
            line.GetPointIds().SetId(1, node_uid_to_vtk_id[member.jnode.uid])
            lines.InsertNextCell(line)
            color = type_color_map.get(member.type, default_color)
            line_colors.InsertNextTuple3(color[0], color[1], color[2])
    
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        poly_data.GetCellData().SetScalars(line_colors)
    
        # --- Create deformed actors for each load combination ---
        deformed_actors = {}
        if loadcombos and scaling:
            displace_scale = scaling.get("displacement", 100)
        
            for combo in loadcombos:
                deformed_points = vtk.vtkPoints()
                for node in self.nodes:
                    if combo.name in node.displacements:
                        deformed_points.InsertNextPoint(
                            node.x_displaced(combo, displace_scale),
                            node.y_displaced(combo, displace_scale),
                            0
                        )
                    else:
                        # If no displacement data for this combo, use original position
                        deformed_points.InsertNextPoint(node.x, node.y, 0)
            
                deformed_poly_data = vtk.vtkPolyData()
                deformed_poly_data.SetPoints(deformed_points)
                deformed_poly_data.SetLines(lines)
            
                deformed_mapper = vtk.vtkPolyDataMapper()
                deformed_mapper.SetInputData(deformed_poly_data)
                deformed_actor = vtk.vtkActor()
                deformed_actor.SetMapper(deformed_mapper)
                deformed_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
                deformed_actor.GetProperty().SetLineStipplePattern(0xF0F0)
                deformed_actor.GetProperty().SetLineWidth(2)
                deformed_actor.SetVisibility(False)  # Initially hidden
            
                deformed_actors[combo.name] = deformed_actor
    
        # --- Create renderer and add all actors ---
        renderer = vtk.vtkRenderer()
    
        # Add base geometry
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(3)
        renderer.AddActor(actor)
    
        # Add all deformed actors
        for actor in deformed_actors.values():
            renderer.AddActor(actor)
    
        # --- Set up the rendering window ---
        renderer.SetBackground(1, 1, 1)
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        # --- Add combo selection interface ---
        # Create combo selector text actors
        combo_text_actors = {}
        if loadcombos:
            for i, combo in enumerate(loadcombos):
                text_actor = vtk.vtkTextActor()
                text_actor.SetInput(f"[{i+1}] {combo.name}")
                text_actor.GetTextProperty().SetColor(0.2, 0.2, 0.8)
                text_actor.GetTextProperty().SetFontSize(14)
                text_actor.SetPosition(10, 10 + i*20)
                combo_text_actors[combo.name] = text_actor
                renderer.AddActor2D(text_actor)
    
            # Create "active combo" indicator
            active_combo_actor = vtk.vtkTextActor()
            active_combo_actor.SetInput("No active combination")
            active_combo_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
            active_combo_actor.GetTextProperty().SetFontSize(16)
            active_combo_actor.GetTextProperty().SetBold(True)
            active_combo_actor.SetPosition(10, 10 + len(loadcombos)*20 + 10)
            renderer.AddActor2D(active_combo_actor)
    

    
        # --- Define Interactor with keyboard controls for combo selection ---
            class MultiComboInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
                def __init__(self, parent=None):
                    self.parent = vtk.vtkRenderWindowInteractor()
                    if parent is not None:
                        self.parent = parent
            
                    self.deformed_actors = deformed_actors
                    self.loadcombos = loadcombos
                    self.active_combo = None
                    self.active_combo_actor = active_combo_actor
                    self.AddObserver("KeyPressEvent", self.key_press_event)
        
                def key_press_event(self, obj, event):
                    key = self.parent.GetKeySym()
            
                    # Handle numeric keys for combo selection
                    if key.isdigit() and int(key) > 0 and int(key) <= len(self.loadcombos):
                        combo_idx = int(key) - 1
                        selected_combo = self.loadcombos[combo_idx]
                
                        # Hide all deformed actors
                        for actor in self.deformed_actors.values():
                            actor.SetVisibility(False)
                
                        # Show only the selected one
                        if selected_combo.name in self.deformed_actors:
                            self.deformed_actors[selected_combo.name].SetVisibility(True)
                            self.active_combo = selected_combo
                            self.active_combo_actor.SetInput(f"Active: {selected_combo.name}")
            
                    elif key == 'h':
                        # Toggle help text
                        pass  # Add help text toggle functionality
            
                    self.parent.GetRenderWindow().Render()
    
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
            interactor.SetInteractorStyle(MultiComboInteractorStyle(parent=interactor))
    
        # --- Start visualization ---
        render_window.Render()
        if loadcombos:
            print("\n--- VTK Interaction ---")
            print("Press keys 1-{} to select different load combinations:".format(len(loadcombos) if loadcombos else 0))
            for i, combo in enumerate(loadcombos or []):
                print(f"  [{i+1}] {combo.name}")
            print("-----------------------\n")
            interactor.Start()