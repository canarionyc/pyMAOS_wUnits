# -*- coding: utf-8 -*-
import os
# from xml.dom import NO_DATA_ALLOWED_ERR
import numpy as np
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal',linewidth= 999)

import ast
import operator

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
    def __init__(self, nodes, members, units=None):
        self.nodes = nodes
        self.members = members
        self.units = units or {}  # Default empty dict if None
        
        # Validate node UIDs are unique
        self._validate_node_uids()
        
        # Validate member UIDs are unique
        self._validate_member_uids()
        
        self.create_uid_maps()  # Replace create_uid_map() with the new method
        
        # Validate that all member nodes exist in the node list
        for member in members:
            if member.inode.uid not in self.uid_to_index:
                raise ValueError(f"Member {member.uid} references node {member.inode.uid} which doesn't exist in the structure")
            if member.jnode.uid not in self.uid_to_index:
                raise ValueError(f"Member {member.uid} references node {member.jnode.uid} which doesn't exist in the structure")
        
        # Rest of initialization remains the same...
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

    def __str__(self):
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
            from pint import UnitRegistry
            ureg = UnitRegistry()
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
        result.append(f"{'Node ID':8} {'X ('+length_unit+')':15} {'Y ('+length_unit+')':15} {'Restraints':15}")
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
        result.append(f"{'Member ID':10} {'Type':8} {'i-node':8} {'j-node':8} {'Length ('+length_unit+')':15}")
        result.append("-" * 80)
        
        for member in sorted(self.members, key=lambda m: m.uid): # type: ignore
            # Convert length to display units
            length_display = Q_(member.length, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else member.length
            
            result.append(f"{member.uid:<10} {member.type:<8} {member.inode.uid:<8} {member.jnode.uid:<8} {length_display:<15.4g}")
    
        # Material properties
        materials_seen = set()
        result.append("\nMATERIAL PROPERTIES:")
        result.append(f"{'Material':10} {'E ('+pressure_unit+')':15}")
        result.append("-" * 80)
        
        for member in self.members:
            if member.material not in materials_seen:
                # Convert elastic modulus to display units
                E_display = Q_(member.material.E, 'Pa').to(pressure_unit).magnitude if hasattr(Q_, 'to') else member.material.E
                
                result.append(f"{member.material.uid:<10} {E_display:<15.4g}")
                materials_seen.add(member.material)
    
        # Section properties
        sections_seen = set()
        result.append("\nSECTION PROPERTIES:")
        result.append(f"{'Section':10} {'Area ('+length_unit+'^2)':18} {'Ixx ('+length_unit+'^4)':18}")
        result.append("-" * 80)
        
        for member in self.members:
            if member.section not in sections_seen:
                # Convert section properties to display units
                area_display = Q_(member.section.Area, 'm^2').to(length_unit+'^2').magnitude if hasattr(Q_, 'to') else member.section.Area
                ixx_display = Q_(member.section.Ixx, 'm^4').to(length_unit+'^4').magnitude if hasattr(Q_, 'to') else member.section.Ixx
                
                result.append(f"{member.section.uid:<10} {area_display:<18.4g} {ixx_display:<18.4g}")
                sections_seen.add(member.section)
    
        # Load information
        has_node_loads = any(node.loads for node in self.nodes)
        if has_node_loads:
            result.append("\nNODAL LOADS:")
            result.append(f"{'Node ID':8} {'Load Case':10} {'Fx ('+force_unit+')':15} {'Fy ('+force_unit+')':15} {'Mz ('+moment_unit+')':15}")
            result.append("-" * 80)
            
            for node in sorted(self.nodes, key=lambda n: n.uid): # type: ignore
                if node.loads:
                    for case, load in node.loads.items():
                        # Convert forces and moments to display units
                        fx_display = Q_(load[0], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[0]
                        fy_display = Q_(load[1], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[1]
                        mz_display = Q_(load[2], 'N*m').to(moment_unit).magnitude if hasattr(Q_, 'to') else load[2]
                        
                        result.append(f"{node.uid:<8} {case:<10} {fx_display:<15.4g} {fy_display:<15.4g} {mz_display:<15.4g}")
    
        # Displacements and reactions if calculated
        has_displacements = any(hasattr(node, 'displacements') and node.displacements for node in self.nodes)
        if has_displacements:
            result.append("\nNODAL DISPLACEMENTS (most recent load combination):")
            result.append(f"{'Node ID':8} {'Ux ('+length_unit+')':15} {'Uy ('+length_unit+')':15} {'Rz (rad)':15}")
            result.append("-" * 80)
            
            for node in sorted(self.nodes, key=lambda n: n.uid): # type: ignore
                if hasattr(node, 'displacements') and node.displacements:
                    # Get the most recent load combination
                    latest_combo = list(node.displacements.keys())[-1]
                    disp = node.displacements[latest_combo]
                    
                    # Convert displacements to display units
                    ux_display = Q_(disp[0], 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else disp[0]
                    uy_display = Q_(disp[1], 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else disp[1]
                    
                    result.append(f"{node.uid:<8} {ux_display:<15.4g} {uy_display:<15.4g} {disp[2]:<15.4g}")
    
        has_reactions = any(hasattr(node, 'reactions') and node.reactions for node in self.nodes)
        if has_reactions:
            result.append("\nSUPPORT REACTIONS (most recent load combination):")
            result.append(f"{'Node ID':8} {'Rx ('+force_unit+')':15} {'Ry ('+force_unit+')':15} {'Mz ('+moment_unit+')':15}")
            result.append("-" * 80)
            
            for node in sorted(self.nodes, key=lambda n: n.uid): # type: ignore
                if hasattr(node, 'reactions') and node.reactions and any(node.restraints):
                    # Get the most recent load combination
                    latest_combo = list(node.reactions.keys())[-1]
                    reaction = node.reactions[latest_combo]
                    
                    # Convert reactions to display units
                    rx_display = Q_(reaction[0], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else reaction[0]
                    ry_display = Q_(reaction[1], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else reaction[1]
                    mz_display = Q_(reaction[2], 'N*m').to(moment_unit).magnitude if hasattr(Q_, 'to') else reaction[2]
                    
                    result.append(f"{node.uid:<8} {rx_display:<15.4g} {ry_display:<15.4g} {mz_display:<15.4g}")
    
        # Warnings and errors
        if self._ERRORS:
            result.append("\nERRORS:")
            for error in self._ERRORS:
                result.append(f"- {error}")
    
        if self._unstable:
            result.append("\nWARNING: Structure is unstable!")
    
        result.append("=" * 80)
        return "\n".join(result)

    def _validate_node_uids(self):
        """
        Validates that all node UIDs are unique.
        Raises a ValueError if duplicate UIDs are found.
        """
        uid_count = {}
        duplicates = []
        
        for node in self.nodes:
            if node.uid in uid_count:
                uid_count[node.uid] += 1
                duplicates.append(node.uid)
            else:
                uid_count[node.uid] = 1
        
        if duplicates:
            duplicate_list = ", ".join(str(uid) for uid in sorted(set(duplicates)))
            raise ValueError(f"Duplicate node UIDs found: {duplicate_list}. Each node must have a unique identifier.")

    def _validate_member_uids(self):
        """
        Validates that all member UIDs are unique.
        Raises a ValueError if duplicate UIDs are found.
        """
        uid_count = {}
        duplicates = []
        
        for member in self.members:
            if member.uid in uid_count:
                uid_count[member.uid] += 1
                duplicates.append(member.uid)
            else:
                uid_count[member.uid] = 1
        
        if duplicates:
            duplicate_list = ", ".join(str(uid) for uid in sorted(set(duplicates)))
            raise ValueError(f"Duplicate member UIDs found: {duplicate_list}. Each member must have a unique identifier.")