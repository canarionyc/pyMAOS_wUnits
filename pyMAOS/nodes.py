# -*- coding: utf-8 -*-
import math


class R2Node:
    def __init__(self, uid, x, y):
        """

        Parameters
        ----------
        x : float
            x-axis coordinate of the node in R2.
        y : float
            y-axis coordinate of the node in R2.
        uid : int
            unique node number.

        Returns
        -------
        None.

        """

        self.uid = uid
        self.x = x
        self.y = y

        # Restraints [Ux, Uy, Rz]
        self.restraints_key = ["Ux", "Uy", "Rz"]
        self.restraints = [0, 0, 0]

        # Spring Restraint [kux, kuy, krz]
        # Spring Stiffness should be 0 for a restrained direction
        # Spring is still a DOF for the node
        self._spring_stiffness = [0, 0, 0]

        # Directionality of spring
        # 0 = bidirectional resistance
        # 1 = spring resists positive axis displacement
        # -1 = spring resists negative axis displacement
        self._spring_direction = [0, 0, 0]

        # Spring Stiffness Multiplier
        # This will be part of Tension/Compression non-linear analysis
        # and will soften or deactivate entirely the spring
        # This needs to be done on a per combination basis so this
        # is a series of dicts for each DOF
        self._springUxmulti = {}
        self._springUymulti = {}
        self._springRzmulti = {}

        # Enforced Displacement [Ux, Uy, Rz]
        # Enforced Displacements count as a restrained DOF if that
        # DOF is not already restrained by a support condition
        self._enforced_displacements = [0, 0, 0]

        # Dict of Loads by case
        self.loads = {}

        # Dict of Displacements by combo
        self.displacements = {}

        # Dict of Reactions by combo
        self.reactions = {}

        # Flags
        self._isSpring = False
        self._isNonLinear = False

    def __str__(self):
        """Return a readable string representation of the node with complete restraint information"""
        restraint_symbols = {0: "Free", 1: "Fixed"}
        
        str_repr = f"Node:{self.uid}\n"
        str_repr += f"Coordinates: ({self.x:.4f}, {self.y:.4f})\n"
        str_repr += "Restraints:\n"
        str_repr += "-" * 15 + "\n"
        
        for i, r in enumerate(self.restraints):
            str_repr += f"{self.restraints_key[i]}: {restraint_symbols[r]}"
            
            # Add spring information if applicable
            if self._isSpring and self._spring_stiffness[i] > 0:
                direction_info = ""
                if self._spring_direction[i] == 1:
                    direction_info = " (+ direction only)"
                elif self._spring_direction[i] == -1:
                    direction_info = " (- direction only)"
                    
                str_repr += f" with Spring k={self._spring_stiffness[i]}{direction_info}"
            
            str_repr += "\n"
            
        return str_repr

    def __repr__(self):
        """Return a string representation of the node for debugging"""
        return f"R2Node(uid={self.uid}, x={self.x}, y={self.y})"

    def x_displaced(self, load_combination, scale=1.0):
        delta = self.displacements.get(load_combination.name, [0, 0, 0])

        return self.x + (delta[0] * scale)

    def y_displaced(self, load_combination, scale=1.0):
        delta = self.displacements.get(load_combination.name, [0, 0, 0])

        return self.y + (delta[1] * scale)

    def distance(self, other):
        """

        Parameters
        ----------
        other : R2Node
            another node defined by this class.

        Returns
        -------
        distance: float
            Euclidean distance in R2

        """

        dx = self.x - other.x
        dy = self.y - other.y

        return math.sqrt((dx * dx) + (dy * dy))

    def restrainUx(self):
        self.restraints[0] = 1

    def restrainUy(self):
        self.restraints[1] = 1

    def restrainMz(self):
        self.restraints[2] = 1

    def restrainAll(self):
        self.restraints = [1, 1, 1]

    def releaseUx(self):
        self.restraints[0] = 0

    def releaseUy(self):
        self.restraints[1] = 0

    def releaseMz(self):
        self.restraints[2] = 0

    def releaseAll(self):
        self.restraints = [0, 0, 0]

    def applySpringUx(self, k=100, direction=0):
        self.restraints[0] = 0
        self._spring_stiffness[0] = k
        self._spring_direction[0] = direction

        self._isSpring = True

        if direction != 0:
            self._isNonLinear = True

    def applySpringUy(self, k=100, direction=0):
        self.restraints[1] = 0
        self._spring_stiffness[1] = k
        self._spring_direction[1] = direction

        self._isSpring = True

        if direction != 0:
            self._isNonLinear = True

    def applySpringRz(self, k=100, direction=0):
        self.restraints[2] = 0
        self._spring_stiffness[2] = k
        self._spring_direction[2] = direction

        self._isSpring = True

        if direction != 0:
            self._isNonLinear = True

    def add_nodal_load(self, fx, fy, mz, loadcase="D"):
        """
        Add a nodal load to the node.

        Parameters
        ----------
        fx : float
            Force in x-direction.
        fy : float
            Force in y-direction.
        mz : float
            Moment around z-axis.
        loadcase : str, optional
            Load case identifier. Default is "D" (Dead Load).
        """
        if loadcase not in self.loads:
            self.loads[loadcase] = [0, 0, 0]
    
        # Add the load components to any existing loads
        self.loads[loadcase][0] += fx
        self.loads[loadcase][1] += fy
        self.loads[loadcase][2] += mz

    def display_loads(self):
        """
        Display all loads applied to the node.
    
        Prints a formatted list of all loads applied to the node,
        organized by load case.
        """
        if not self.loads:
            print(f"Node:{self.uid} - No loads applied")
            return
    
        print(f"Node:{self.uid} - Applied Loads")
        print("-" * 20)
    
        for loadcase, forces in self.loads.items():
            print(f"Load Case: {loadcase}")
            print(f"  Fx: {forces[0]:.4E}, Fy: {forces[1]:.4E}, Mz: {forces[2]:.4E}")

# --- Read nodes ---
def get_nodes_from_csv(csv_file):
    import csv
    nodes_dict = {}
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            uid = int(row["uid"])
            x = float(row["X"])
            y = float(row["Y"])
            rx = int(row["rx"])
            ry = int(row["ry"])
            rz = int(row["rz"])
            node = R2Node(uid, x, y)
            node.restraints = [rx, ry, rz]
            nodes_dict[uid] = node
    return nodes_dict