# -*- coding: utf-8 -*-
import math
import numpy as np
import pint
from pint import Quantity
import pyMAOS
from pyMAOS.pymaos_units import parse_value_with_units


class LoadsDict(dict):
    """Custom dictionary class that handles unit parsing for load values"""

    def __init__(self, node):
        super().__init__()
        self.node = node  # Reference to the parent node for access to parsing methods

    def __setitem__(self, key, value):
        """Override setitem to parse units when assigning individual load cases"""
        if isinstance(value, (list, tuple)):
            # Parse the load values with units
            parsed_values = self.node._parse_load_values(value)
            super().__setitem__(key, parsed_values)
        else:
            super().__setitem__(key, value)

    def update(self, other_dict):
        """Override update to parse units for bulk updates"""
        if isinstance(other_dict, dict):
            for key, value in other_dict.items():
                self[key] = value  # Use our custom __setitem__


class R2Node:
    def __init__(self, uid, x, y):
        """
        Parameters
        ----------
        x : float, str, or pint.Quantity
            x-axis coordinate of the node in R2.
        y : float, str, or pint.Quantity
            y-axis coordinate of the node in R2.
        uid : int
            unique node number.
        """
        self.uid = uid
        # Ux,Uy, and Rz = 3
        # Number of possible Joint Displacements
        self.NJD = 3

        self.is_inode_of_elem_ids = []; self.is_jnode_of_elem_ids = []

        self.is_inode_of_elems=[]; self.is_jnode_of_elems=[]

        # Parse and store coordinates as pint.Quantity objects
        self.x = self._parse_coordinate(x)
        self.y = self._parse_coordinate(y)

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
        self._springUxmulti = {}
        self._springUymulti = {}
        self._springRzmulti = {}

        # Enforced Displacement [Ux, Uy, Rz]
        self._enforced_displacements = [0, 0, 0]

        # Dict of Loads by case - use custom dictionary that handles unit parsing
        self.loads = LoadsDict(self)

        # Dict of Displacements by combo
        self.displacements = {}

        # Dict of Reactions by combo
        self.reactions = {}

        # Flags
        self._isSpring = False
        self._isNonLinear = False

    def _parse_coordinate(self, coordinate):
        # If already a Quantity, convert to internal length units
        if isinstance(coordinate, pint.Quantity):
            return coordinate.to(pyMAOS.INTERNAL_LENGTH_UNIT)

        # If it's a number, create a Quantity in internal length units
        if isinstance(coordinate, (int, float)):
            return coordinate * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_LENGTH_UNIT)

        # If it's a string, parse units
        if isinstance(coordinate, str):
            try:
                parsed_value = parse_value_with_units(coordinate)

                if isinstance(parsed_value, pint.Quantity):
                    return parsed_value.to(pyMAOS.INTERNAL_LENGTH_UNIT)
                else:
                    # No units specified, assume internal length units
                    return float(parsed_value) * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_LENGTH_UNIT)

            except Exception as e:
                print(f"Warning: Could not parse coordinate '{coordinate}': {e}")
                try:
                    return float(coordinate) * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_LENGTH_UNIT)
                except Exception:
                    raise ValueError(f"Could not parse coordinate value: {coordinate}")

        raise ValueError(f"Unsupported coordinate type: {type(coordinate)}")

    def _parse_load_value(self, load_value):
        # If already a Quantity, convert to appropriate internal units
        if isinstance(load_value, pint.Quantity):
            # Check dimensionality to convert to correct units
            if load_value.dimensionality == pyMAOS.FORCE_DIMENSIONALITY:
                return load_value.to(pyMAOS.INTERNAL_FORCE_UNIT)
            elif load_value.dimensionality == pyMAOS.MOMENT_DIMENSIONALITY:
                return load_value.to(pyMAOS.INTERNAL_MOMENT_UNIT)
            return load_value

        # If it's a number, assume internal force units
        if isinstance(load_value, (int, float)):
            return float(load_value) * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_FORCE_UNIT)

        # If it's a string, parse units
        if isinstance(load_value, str):
            try:
                parsed_value = parse_value_with_units(load_value)

                if isinstance(parsed_value, pint.Quantity):
                    # Convert to appropriate internal units based on dimensionality
                    if parsed_value.dimensionality == pyMAOS.FORCE_DIMENSIONALITY:
                        return parsed_value.to(pyMAOS.INTERNAL_FORCE_UNIT)
                    elif parsed_value.dimensionality == pyMAOS.MOMENT_DIMENSIONALITY:
                        return parsed_value.to(pyMAOS.INTERNAL_MOMENT_UNIT)
                    return parsed_value
                else:
                    # No units specified, assume internal force units
                    return float(parsed_value) * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_FORCE_UNIT)

            except Exception as e:
                print(f"Warning: Could not parse load value '{load_value}': {e}")
                try:
                    return float(load_value) * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_FORCE_UNIT)
                except Exception:
                    raise ValueError(f"Could not parse load value: {load_value}")

        raise ValueError(f"Unsupported load value type: {type(load_value)}")

    def _parse_load_values(self, load_values):
        """
        Parse a list of load values [fx, fy, mz].

        Parameters
        ----------
        load_values : list
            List of load values [fx, fy, mz]

        Returns
        -------
        list
            List of load values as pint.Quantity objects
        """
        if not isinstance(load_values, (list, tuple)):
            raise ValueError("Load values must be a list or tuple")

        if len(load_values) != 3:
            raise ValueError("Load values must contain exactly 3 elements [fx, fy, mz]")

        parsed_loads = []
        for i, load_value in enumerate(load_values):
            parsed_loads.append(self._parse_load_value(load_value))

        return parsed_loads

    def __str__(self):
        """Return a readable string representation of the node with complete restraint information"""
        restraint_symbols = {0: "Free", 1: "Fixed"}

        str_repr = f"Node:{self.uid}\n"
        str_repr += f"Coordinates: ({self.x}, {self.y})\n"
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

    def x_displaced(self, combo, scale=1.0):
        """Return X coordinate with displacement applied"""
        if combo.name in self.displacements:
            # Get displacement in compatible form
            raw_disp = self.displacements[combo.name][0]
            
            # Handle case where displacement is stored as a plain number
            if not hasattr(raw_disp, 'to'):
                # Convert to a quantity with appropriate units
                raw_disp = raw_disp * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_LENGTH_UNIT)
            
            # Convert the displacement to the same unit as self.x
            disp_in_x_units = raw_disp.to(self.x.units) * scale
            
            # Add using the same registry
            return self.x + disp_in_x_units
        return self.x

    def y_displaced(self, combo, scale=1.0):
        """Return Y coordinate with displacement applied"""
        if combo.name in self.displacements:
            # Get displacement in compatible form
            raw_disp = self.displacements[combo.name][1]
            
            # Handle case where displacement is stored as a plain number
            if not hasattr(raw_disp, 'to'):
                # Convert to a quantity with appropriate units
                raw_disp = raw_disp * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_LENGTH_UNIT)
            
            # Convert the displacement to the same unit as self.y
            disp_in_y_units = raw_disp.to(self.y.units) * scale
            
            # Add using the same registry
            return self.y + disp_in_y_units
        return self.y

    def z_rotated(self, combo, scale=1.0):
        """Return Z rotation angle with displacement applied"""
        if combo.name in self.displacements:
            raw_rot = self.displacements[combo.name][2]
            # Handle case where rotation is stored as a plain number
            if not hasattr(raw_rot, 'magnitude'):
                # Convert to a quantity with appropriate units
                return raw_rot * scale * pyMAOS.unit_manager.ureg('rad')
            return raw_rot * scale
        return 0 * pyMAOS.unit_manager.ureg('rad')

    def distance(self, other):
        """
        Calculate Euclidean distance between nodes.

        Parameters
        ----------
        other : R2Node
            Another node

        Returns
        -------
        pint.Quantity
            Euclidean distance in meters
        """
        dx = self.x - other.x
        dy = self.y - other.y

        # Calculate distance preserving units
        return (dx ** 2 + dy ** 2) ** 0.5

    def restrainUx(self):
        self.restraints[0] = 1

    def restrainUy(self):
        self.restraints[1] = 1

    def restrainMz(self):
        self.restraints[2] = 1

    def restrainAll(self):
        self.restraints = [1, 1, 1]

    def restrainTranslation(self):
        self.restraints = [1, 1, 0]

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
        if loadcase not in self.loads:
            zero_force = 0 * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_FORCE_UNIT)
            zero_moment = 0 * pyMAOS.unit_manager.ureg(pyMAOS.INTERNAL_MOMENT_UNIT)
            self.loads[loadcase] = [zero_force, zero_force, zero_moment]

        # Parse load values with units
        fx_parsed = self._parse_load_value(fx)
        fy_parsed = self._parse_load_value(fy)
        mz_parsed = self._parse_load_value(mz)

        # Add the load components to any existing loads
        self.loads[loadcase][0] += fx_parsed
        self.loads[loadcase][1] += fy_parsed
        self.loads[loadcase][2] += mz_parsed

    def display_loads(self):
        """Display all loads applied to the node."""
        if not self.loads:
            print(f"Node:{self.uid} - No loads applied")
            return

        print(f"Node:{self.uid} - Applied Loads")
        print("-" * 50)

        for loadcase, forces in self.loads.items():
            print(f"Load Case: {loadcase}")
            print(f"  Fx: {forces[0]}, Fy: {forces[1]}, Mz: {forces[2]}")

    def compute_reactions(self, load_combination):
        """Calculate reactions for a given load combination

        Computes support reactions by summing member end forces
        at support nodes.

        Parameters
        ----------
        load_combination : LoadCombo
            The load combination for which to calculate reactions

        Returns
        -------
        numpy.ndarray
            Vector of reaction forces/moments at support nodes
        """
        # Initialize reactions array with object dtype to store quantities with units
        reactions = np.zeros(self.NJD, dtype=object)

        # Convert zeros to quantities with appropriate units
        import pyMAOS
        for i in range(self.NJD):
            # Use alternating force and moment units based on DOF index
            unit = pyMAOS.unit_manager.get_current_units().get('force' if i % 3 != 2 else 'moment', 'N')
            reactions[i] = pyMAOS.unit_manager.get_zero_quantity(unit)

        print(f"Computing reactions for load combination: {load_combination.name}")

        # For each member, add its contribution to the reactions
        from pyMAOS.quantity_utils import add_arrays_with_units
        # First iterate through members where this node is the i-node
        for member in self.is_inode_of_elems:
            member_FG = member.set_end_forces_global(load_combination)
            # Use slice for i-node forces (first part of force vector)
            reactions = add_arrays_with_units(reactions, member_FG[0:self.NJD])
            print(f"After adding i-node member {member.uid}, reactions = {reactions}")

        # Then iterate through members where this node is the j-node
        for member in self.is_jnode_of_elems:
            member_FG = member.set_end_forces_global(load_combination)
            # Use slice for j-node forces (second part of force vector)
            reactions = add_arrays_with_units(reactions, member_FG[self.NJD:(2*self.NJD)])
            print(f"After adding j-node member {member.uid}, reactions = {reactions}")

        self.reactions = reactions
        print(f"Node {self.uid} reactions: Rx={reactions[0]:.4E}, Ry={reactions[1]:.4E}, Mz={reactions[2]:.4E}")

        return reactions

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