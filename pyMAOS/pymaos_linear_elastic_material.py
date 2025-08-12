from xarray.coding.times import resolve_time_unit_from_attrs_dtype
import os
import pyMAOS

from typing import Union
import pint
from pint import Quantity
from pyMAOS import unit_manager, info, debug, error
from quantity_utils import convert_registry
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

        # Get internal units directly from unit_manager (force refresh)
        import pyMAOS

        # Debug the unit manager state
        # info(f"Unit manager system: {pyMAOS.unit_manager.system_name}")
        # info(f"Unit manager base units: Force={pyMAOS.unit_manager.INTERNAL_FORCE_UNIT}, Length={pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT}")

        # Force update derived units before accessing them
        # pyMAOS.unit_manager._update_derived_units()

        # Now get the updated internal units
        internal_pressure_unit = pyMAOS.unit_manager.INTERNAL_PRESSURE_UNIT
        internal_density_unit = pyMAOS.unit_manager.INTERNAL_DENSITY_UNIT
        system_name = pyMAOS.unit_manager.system_name


        # info(f"Using internal unit system: {system_name}")
        # info(f"Internal pressure unit: {internal_pressure_unit}")
        # info(f"Internal pressure unit expanded: {pyMAOS.unit_manager.INTERNAL_PRESSURE_UNIT_EXPANDED}")
        # info(f"Internal density unit: {internal_density_unit}")

        # Process Young's modulus
        if isinstance(E, (int, float)):
            raise TypeError("Young's modulus E must be provided as a string with units or a Quantity object")

        if isinstance(E, str):
            E = unit_manager.ureg.Quantity(E)

        # Validate dimensions for Young's modulus (pressure)
        if not E.check('[pressure]'):
            raise pint.DimensionalityError(f"Young's modulus E has incorrect dimensions: {E.dimensionality}. "
                             f"Expected dimensions: [length]^-1 [mass]^1 [time]^-2")

        self.E = E.to(internal_pressure_unit)

        # Process density
        if isinstance(density, (int, float)):
            raise TypeError("Density must be provided as a string with units or a Quantity object")

        if  isinstance(density, str):
            density = unit_manager.ureg.Quantity(density)


        # Validate dimensions for density
        if not density.check('[density]'):
            raise pint.DimensionalityError(f"Density has incorrect dimensions: {density.dimensionality}. "
                             f"Expected dimensions: [length]^-3 [mass]^1")

        self.density = density.to(internal_density_unit)

        print(f"LinearElasticMaterial uid {self.uid} initialized with "
              f"density={self.density:.4g}, E={self.E:.3g}, nu={self.nu}")


    def __setstate__(self, state: dict[str, any]) -> None:
        """
        Restore the object state during unpickling.

        Parameters
        ----------
        state : dict
            Dictionary containing the object state
        """
        # Handle E value properly based on its type
        e_value = state.get('E', "30000 ksi")
        if isinstance(e_value, (int, float)):
            pyMAOS.warning(f"assuming E {e_value} is in {unit_manager.INTERNAL_PRESSURE_UNIT}")
            e_quantity = unit_manager.ureg.Quantity(e_value, unit_manager.INTERNAL_PRESSURE_UNIT)
        elif isinstance(e_value, pint.Quantity):
            # If it's already a Quantity object, use it directly
            e_quantity = convert_registry(e_value, unit_manager.ureg)
        else:
            # If it's a string, parse it with ureg
            e_quantity = unit_manager.ureg(e_value)

        # Handle density value properly based on its type
        density_value = state.get('density', "0.284 lb/in**3")
        if isinstance(density_value, (int, float)):
            pyMAOS.warning(f"assuming density {density_value} is in {unit_manager.INTERNAL_DENSITY_UNIT}")
            density_quantity = unit_manager.ureg.Quantity(density_value, unit_manager.INTERNAL_DENSITY_UNIT)
        elif isinstance(density_value, pint.Quantity):
            # If it's already a Quantity object, use it directly
            density_quantity = convert_registry(density_value, unit_manager.ureg)
        else:
            # If it's a string, parse it with ureg
            density_quantity = unit_manager.ureg(density_value)

        # Pass the values directly to __init__ for validation and processing
        self.__init__(
            uid=state.get('uid', 0),
            density=density_quantity,
            E=e_quantity,
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
    Loads materials from a YAML file with proper object deserialization

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
    from pyMAOS import info

    info(f"Loading materials from: {materials_yml}")
    assert os.path.exists(materials_yml), f"{materials_yml} not found"
    # Load YAML file with object deserialization
    with open(materials_yml, 'r') as file:
        import yaml
        materials_list = yaml.unsafe_load(file)

    # Convert list to dictionary keyed by UID
    materials_dict = {material.uid: material for material in materials_list}
    print(materials_dict)
    info(f"Loaded {len(materials_dict)} materials")
    return materials_dict