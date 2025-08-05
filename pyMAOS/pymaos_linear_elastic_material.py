from xarray.coding.times import resolve_time_unit_from_attrs_dtype

import pyMAOS

from typing import Union
import pint
from pint import Quantity
from pyMAOS import unit_manager

class LinearElasticMaterial():
    def __init__(self, uid: int,
             density: Union[str, Quantity],
             E: Union[str, Quantity],
             nu: float = 0.3):
        """
        Initialize a linear elastic material with proper unit validation.

        Parameters
        ----------
        uid : int
            Unique Material Identifier
        density : Union[str, Quantity]
            Material density with units (e.g. "7850 kg/m^3")
        E : Union[str, Quantity]
            Young's modulus with units (e.g. "210 GPa")
        nu : float
            Poisson's ratio (dimensionless)
        """
        super().__init__()
        self.uid = uid
        self.nu = nu

        # Process Young's modulus
        if isinstance(E, (int, float)):
            raise TypeError("Young's modulus E must be provided as a string with units or a Quantity object")

        if not isinstance(E, pint.Quantity):
            E = unit_manager.ureg(E)

        # Validate dimensions for Young's modulus (pressure)
        if not E.check({'[length]': -1, '[mass]': 1, '[time]': -2}):
            raise ValueError(f"Young's modulus E has incorrect dimensions: {E.dimensionality}. "
                             f"Expected dimensions: [length]^-1 [mass]^1 [time]^-2")

        self.E = E.to_reduced_units()

        # Process density
        if isinstance(density, (int, float)):
            raise TypeError("Density must be provided as a string with units or a Quantity object")

        if not isinstance(density, pint.Quantity):
            density = unit_manager.ureg(density)

        # Validate dimensions for density
        if not density.check({'[length]': -3, '[mass]': 1}):
            raise ValueError(f"Density has incorrect dimensions: {density.dimensionality}. "
                             f"Expected dimensions: [length]^-3 [mass]^1")

        self.density = density.to_reduced_units()

        print(f"LinearElasticMaterial uid {self.uid} initialized with "
              f"density={self.density:.4g}, E={self.E:.3g}, nu={self.nu}")


    def __setstate__(self, state):
        """
        Restore the object state during unpickling.

        Parameters
        ----------
        state : dict
            Dictionary containing the object state
        """
        # Pass the values directly to __init__ for validation and processing
        self.__init__(
            uid=state.get('uid', 0),
            density=state.get('density', "0 kg/m^3"),  # Let __init__ handle validation
            E=state.get('E', "0 Pa"),                  # Let __init__ handle validation
            nu=state.get('nu', 0.3)
        )

    def _parse_value_with_units(self, value: Union[float, str], unit_type: str) -> float:
        """
        Parse a value that may be a float or string with units.
        
        Parameters
        ----------
        value : float or str
            Value, either as a number or string with units (e.g., "29000ksi", "0.000284klb/in^3")
        unit_type : str
            Type of unit ('pressure', 'density')
            
        Returns
        -------
        float
            Value in SI units
        """
        # If it's already a number, return it as-is (assume SI units)
        if isinstance(value, (int, float)):
            return float(value)

        # If it's a string, try to parse units
        if isinstance(value, str):
            try:
                # Import the parse function from units_mod
                from pyMAOS.pymaos_units import parse_value_with_units
                import pint

                # Parse the value string
                parsed_value = parse_value_with_units(value)

                # If it has units, convert to SI units
                if isinstance(parsed_value, pint.Quantity):

                    # Convert based on unit type
                    if unit_type == 'pressure':
                        # Young's modulus - convert to Pascals
                        si_value = parsed_value.to('Pa')#.magnitude
                    elif unit_type == 'density':
                        # Density - convert to kg/m^3
                        si_value = parsed_value.to('kg/m^3')#.magnitude
                    else:
                        # Fallback - just use the magnitude
                        si_value = None
                    return float(si_value)
            except Exception as e:
                print(f"Warning: Could not convert '{value}' to SI units for {unit_type}: {e}")
                # Fall back to magnitude if conversion fails
                return

    # If we get here, something unexpected happened
        raise ValueError(f"Unsupported material value type: {type(value)}")

    def stress(self, strain):
        return self.E * strain

    def __str__(self):
        """Return string representation of the material properties"""

        units = pyMAOS.unit_manager.get_current_units()
        e_display = self.E.to(units['pressure'])
        density_display = self.density.to(units['density'])
        # pressure_unit = units.get('pressure', 'Pa')
        # density_unit = units.get('density', 'kg/m^3')
        return f"LinearElasticMaterial(uid={self.uid}, density={density_display:.4f}, E={e_display}, nu={self.nu}) in {units.get('name', 'default')} units"

    def __repr__(self):
        """Return developer representation of the material"""
        return f"LinearElasticMaterial(uid={self.uid}, density={self.density:.4f}, E={self.E:.2f}, nu={self.nu})"

def get_materials_from_yaml(materials_yml, logger=None):
    """
    Loads materials from a YAML file and converts properties to internal units

    Parameters
    ----------
    materials_yml : str
        Path to the materials YAML file
    logger : logging.Logger, optional
        Logger for output messages

    Returns
    -------
    dict
        Dictionary of materials with uid as key
    """
    materials_dict = {}

    # Use print or logger.info based on what's available
    def log(message):
        if logger:
            logger.info(message)
        else:
            print(message)

    # Get internal units directly from unit_manager (force refresh)
    import pyMAOS

    # Debug the unit manager state
    log(f"Unit manager system: {pyMAOS.unit_manager.system_name}")
    log(f"Unit manager base units: Force={pyMAOS.unit_manager.INTERNAL_FORCE_UNIT}, Length={pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT}")

    # Force update derived units before accessing them
    # pyMAOS.unit_manager._update_derived_units()

    # Now get the updated internal units
    internal_pressure_unit = pyMAOS.unit_manager.INTERNAL_PRESSURE_UNIT
    internal_density_unit = pyMAOS.unit_manager.INTERNAL_DENSITY_UNIT
    system_name = pyMAOS.unit_manager.system_name

    log(f"Loading materials from: {materials_yml}")
    log(f"Using internal unit system: {system_name}")
    log(f"Internal pressure unit: {internal_pressure_unit}")
    log(f"Internal pressure unit expanded: {pyMAOS.unit_manager.INTERNAL_PRESSURE_UNIT_EXPANDED}")
    log(f"Internal density unit: {internal_density_unit}")

    # Rest of the function...

    with open(materials_yml, 'r') as file:
        import yaml
        materials_list = yaml.unsafe_load(file)

    # Process each material and convert units to internal units
    for material in materials_list:
        # If the material is already a LinearElasticMaterial instance, update its units
        if hasattr(material, 'E') and hasattr(material, 'density'):
            # Convert E to internal pressure units if it's a Quantity
            import pint
            if isinstance(material.E, pint.Quantity):
                material.E = material.E.to(internal_pressure_unit)

            # Convert density to internal density units if it's a Quantity
            if isinstance(material.density, pint.Quantity):
                material.density = material.density.to(internal_density_unit)

            log(f"Material {material.uid}: E={material.E}, density={material.density}")

        # Add material to dictionary
        materials_dict[material.uid] = material

    log(f"Loaded {len(materials_dict)} materials")
    return materials_dict