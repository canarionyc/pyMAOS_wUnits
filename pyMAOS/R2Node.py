# -*- coding: utf-8 -*-
import math


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
        x : float or str
            x-axis coordinate of the node in R2. Can be a float or string with units (e.g., "120in").
            If units are provided, they will be converted to SI units (meters).
        y : float or str
            y-axis coordinate of the node in R2. Can be a float or string with units (e.g., "10ft").
            If units are provided, they will be converted to SI units (meters).
        uid : int
            unique node number.

        Returns
        -------
        None.

        """

        self.uid = uid
        
        # Parse and convert coordinates to SI units (meters)
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
        """
        Parse a coordinate value that may be a float or string with units.
        
        Parameters
        ----------
        coordinate : float or str
            Coordinate value, either as a number or string with units (e.g., "120in", "10ft")
            
        Returns
        -------
        float
            Coordinate value in SI units (meters)
        """
        # If it's already a number, return it as-is (assume SI units)
        if isinstance(coordinate, (int, float)):
            return float(coordinate)
        
        # If it's a string, try to parse units
        if isinstance(coordinate, str):
            try:
                # Import the parse function from units_mod
                from pyMAOS.units_mod import parse_value_with_units
                import pint
                
                # Parse the coordinate string
                parsed_value = parse_value_with_units(coordinate)
                
                # If it has units, convert to meters
                if isinstance(parsed_value, pint.Quantity):
                    try:
                        meters_value = parsed_value.to('m').magnitude
                        return float(meters_value)
                    except Exception as e:
                        print(f"Warning: Could not convert '{coordinate}' to meters: {e}")
                        # Fall back to magnitude if conversion fails
                        return float(parsed_value.magnitude)
                else:
                    # No units, just return the numeric value
                    return float(parsed_value)
                    
            except Exception as e:
                print(f"Warning: Could not parse coordinate '{coordinate}': {e}")
                # Try to convert directly to float as fallback
                try:
                    return float(coordinate)
                except Exception:
                    raise ValueError(f"Could not parse coordinate value: {coordinate}")
        
        # If we get here, something unexpected happened
        raise ValueError(f"Unsupported coordinate type: {type(coordinate)}")

    def _parse_load_value(self, load_value):
        """
        Parse a load value that may be a float or string with units.
    
        Parameters
        ----------
        load_value : float or str
            Load value, either as a number or string with units (e.g., "50kip", "100kN")
        
        Returns
        -------
        float
            Load value in SI units (Newtons for forces, Newton-meters for moments)
        """
      # If it's already a number, return it as-is (assume SI units)
        if isinstance(load_value, (int, float)):
            return float(load_value)
    
        # If it's a string, try to parse units
        if isinstance(load_value, str):
            try:
                # Import from units_mod
                from pyMAOS.units_mod import parse_value_with_units, convert_to_internal_units
                from pyMAOS.units_mod import FORCE_DIMENSIONALITY, MOMENT_DIMENSIONALITY
            
                # Parse the value string into a quantity with units
                parsed_value = parse_value_with_units(load_value)
            
                # Check dimensionality and convert to appropriate SI unit
                if hasattr(parsed_value, 'dimensionality'):
                    # Efficiently check against pre-computed dimensionality constants
                    if parsed_value.dimensionality == FORCE_DIMENSIONALITY:
                        return convert_to_internal_units(parsed_value, 'force')
                    elif parsed_value.dimensionality == MOMENT_DIMENSIONALITY:
                        return convert_to_internal_units(parsed_value, 'moment')
                    else:
                        # Unknown dimension - return magnitude as fallback
                        print(f"Warning: Unknown dimensionality in '{load_value}'")
                        return float(parsed_value.magnitude)
                else:
                    # No dimensionality - return as is
                    return float(parsed_value)
                
            except Exception as e:
                print(f"Warning: Could not parse load value '{load_value}': {e}")
                # Try to convert directly to float as fallback
                try:
                    return float(load_value)
                except Exception:
                    raise ValueError(f"Could not parse load value: {load_value}")
    
        # If we get here, something unexpected happened
        raise ValueError(f"Unsupported load value type: {type(load_value)}")

    def _parse_load_values(self, load_values):
        """
        Parse a list of load values [fx, fy, mz] that may contain strings with units.
        
        Parameters
        ----------
        load_values : list
            List of load values [fx, fy, mz]
            
        Returns
        -------
        list
            List of load values in SI units
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
        str_repr += f"Coordinates: ({self.x:.4f}, {self.y:.4f}) [SI units: meters]\n"
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

    # In your Node class
    def x_displaced(self, combo, scale=1.0):
        """Return X coordinate with displacement applied"""
        if combo.name in self.displacements:
            return self.x + self.displacements[combo.name][0] * scale
        return self.x

    def y_displaced(self, combo, scale=1.0):
        """Return Y coordinate with displacement applied"""
        if combo.name in self.displacements:
            return self.y + self.displacements[combo.name][1] * scale
        return self.y

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
        """
        Add a nodal load to the node.

        Parameters
        ----------
        fx : float or str
            Force in x-direction. Can be numeric or string with units (e.g., "50kip").
        fy : float or str
            Force in y-direction. Can be numeric or string with units (e.g., "100kN").
        mz : float or str
            Moment around z-axis. Can be numeric or string with units (e.g., "500kip*ft").
        loadcase : str, optional
            Load case identifier. Default is "D" (Dead Load).
        """
        if loadcase not in self.loads:
            self.loads[loadcase] = [0, 0, 0]
    
        # Parse load values with units
        fx_parsed = self._parse_load_value(fx)
        fy_parsed = self._parse_load_value(fy)
        mz_parsed = self._parse_load_value(mz)
    
        # Add the load components to any existing loads
        self.loads[loadcase][0] += fx_parsed
        self.loads[loadcase][1] += fy_parsed
        self.loads[loadcase][2] += mz_parsed

    def display_loads(self):
        """
        Display all loads applied to the node.
    
        Prints a formatted list of all loads applied to the node,
        organized by load case. All values shown in SI units.
        """
        if not self.loads:
            print(f"Node:{self.uid} - No loads applied")
            return
    
        print(f"Node:{self.uid} - Applied Loads [SI units: N, N*m]")
        print("-" * 50)
    
        for loadcase, forces in self.loads.items():
            print(f"Load Case: {loadcase}")
            print(f"  Fx: {forces[0]:.4E} N, Fy: {forces[1]:.4E} N, Mz: {forces[2]:.4E} N*m")

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