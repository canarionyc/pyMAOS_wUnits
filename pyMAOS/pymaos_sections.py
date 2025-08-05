import numpy as np
import pyMAOS
from pyMAOS.unit_aware import UnitAwareMixin

from typing import Union
import pint
from pint import Quantity
from math import isnan

class Section():
    def __init__(self, uid: int,
                 Area: Union[float, str, Quantity] = 10.0,
                 Ixx: Union[float, str, Quantity] = np.nan,
                 Iyy: Union[float, str, Quantity] = np.nan):
        super().__init__()
        self.uid = uid

        # Parse and store values with proper unit handling
        self.Area = self._parse_value_with_units(Area, 'area')
        self.Ixx = self._parse_value_with_units(Ixx, 'moment_of_inertia')
        self.Iyy = self._parse_value_with_units(Iyy, 'moment_of_inertia')

    def __setstate__(self, state):
        from pint import UnitRegistry
        

        # Get values from state
        uid = state.get('uid')
        area = state.get('area', "10 in^2")
        ixx_value = state.get('ixx', np.nan)
        iyy_value = state.get('iyy', np.nan)

        # Convert to Quantity objects or leave as NaN
        area_qty = pyMAOS.unit_manager.ureg(area)
        ixx_qty = pyMAOS.unit_manager.ureg(ixx_value) if not isinstance(ixx_value, float) or not np.isnan(ixx_value) else np.nan
        iyy_qty = pyMAOS.unit_manager.ureg(iyy_value) if not isinstance(iyy_value, float) or not np.isnan(iyy_value) else np.nan

        # Initialize with processed values
        self.__init__(uid, area_qty, ixx_qty, iyy_qty)

    @staticmethod
    def _parse_value_with_units(value: Union[float, str, Quantity], unit_type: str) -> Quantity:
        """
        Parse a value that may be a float, string with units, or Quantity.

        Returns
        -------
        pint.Quantity
            Value with appropriate units
        """
        # Handle NaN case
        if isinstance(value, (int, float)) and np.isnan(float(value)):
            return np.nan

        # Get appropriate internal units based on type
        internal_area_unit = pyMAOS.unit_manager.INTERNAL_AREA_UNIT
        internal_inertia_unit = pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT

        # If it's already a Quantity, convert to appropriate internal units
        import pint
        if isinstance(value, pint.Quantity):
            if unit_type == 'area':
                return value.to(internal_area_unit)
            elif unit_type == 'moment_of_inertia':
                return value.to(internal_inertia_unit)
            return value

        # If it's a number, use internal units
        if isinstance(value, (int, float)):
            if unit_type == 'area':
                return value * pyMAOS.unit_manager.ureg(internal_area_unit)
            elif unit_type == 'moment_of_inertia':
                return value * pyMAOS.unit_manager.ureg(internal_inertia_unit)
            return value * pyMAOS.unit_manager.ureg.dimensionless

        # If it's a string, parse units
        if isinstance(value, str):
            try:
                from pyMAOS.pymaos_units import parse_value_with_units
                parsed_value = parse_value_with_units(value)

                if isinstance(parsed_value, pint.Quantity):
                    if unit_type == 'area':
                        return parsed_value.to(internal_area_unit)
                    elif unit_type == 'moment_of_inertia':
                        return parsed_value.to(internal_inertia_unit)
                    return parsed_value
                else:
                    # No units in string, create with internal units
                    if unit_type == 'area':
                        return parsed_value * pyMAOS.unit_manager.ureg(internal_area_unit)
                    elif unit_type == 'moment_of_inertia':
                        return parsed_value * pyMAOS.unit_manager.ureg(internal_inertia_unit)
                    return parsed_value * pyMAOS.unit_manager.ureg.dimensionless

            except Exception as e:
                print(f"Warning: Could not parse section value '{value}': {e}")
                try:
                    num_value = float(value)
                    if unit_type == 'area':
                        return num_value * pyMAOS.unit_manager.ureg(internal_area_unit)
                    elif unit_type == 'moment_of_inertia':
                        return num_value * pyMAOS.unit_manager.ureg(internal_inertia_unit)
                    return num_value * pyMAOS.unit_manager.ureg.dimensionless
                except:
                    raise ValueError(f"Could not parse section value: {value}")

        raise ValueError(f"Unsupported section value type: {type(value)}")

    def __str__(self):
        """Return string representation of the section properties"""
        units = pyMAOS.unit_manager.get_current_units()
        area_display = self.Area.to(units['area'])

        # Handle possible NaN values for moments of inertia
        if isinstance(self.Ixx, (int, float)) and np.isnan(self.Ixx):
            ixx_display = "NaN"
        else:
            ixx_display = self.Ixx.to(units['moment_of_inertia'])

        if isinstance(self.Iyy, (int, float)) and np.isnan(self.Iyy):
            iyy_display = "NaN"
        else:
            iyy_display = self.Iyy.to(units['moment_of_inertia'])

        return f"Section {self.uid}: A={area_display:.2f}, Ixx={ixx_display:.2f}, Iyy={iyy_display}"

    def __repr__(self):
        """Return developer representation of the section"""
        return f"Section(uid={self.uid}, Area={self.Area}, Ixx={self.Ixx}, Iyy={self.Iyy})"

def get_sections_from_yaml(sections_yml, logger=None):
    """
    Loads sections from a YAML file and converts properties to internal units

    Parameters
    ----------
    sections_yml : str
        Path to the sections YAML file
    logger : logging.Logger, optional
        Logger for output messages

    Returns
    -------
    dict
        Dictionary of sections with uid as key
    """
    sections_dict = {}

    # Use print or logger.info based on what's available
    def log(message):
        if logger:
            logger.info(message)
        else:
            print(message)

    # Get internal units from unit_manager
    import pyMAOS
    internal_area_unit = pyMAOS.unit_manager.INTERNAL_AREA_UNIT
    internal_inertia_unit = pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT
    system_name = pyMAOS.unit_manager.system_name

    log(f"Loading sections from: {sections_yml}")
    log(f"Using internal unit system: {system_name}")
    log(f"Internal area unit: {internal_area_unit}")
    log(f"Internal inertia unit: {internal_inertia_unit}")

    with open(sections_yml, 'r') as file:
        import yaml
        sections_list = yaml.unsafe_load(file)

    # Process each section and convert units to internal units
    for section in sections_list:
        # If the section is already a Section instance, update its units
        import pint
        if hasattr(section, 'Area'):
            # Convert Area to internal area units if it's a Quantity
            if isinstance(section.Area, pint.Quantity):
                section.Area = section.Area.to(internal_area_unit)

            # Convert Ixx to internal inertia units if it's a Quantity
            if isinstance(section.Ixx, pint.Quantity) and not np.isnan(section.Ixx):
                section.Ixx = section.Ixx.to(internal_inertia_unit)

            # Convert Iyy to internal inertia units if it's a Quantity
            if isinstance(section.Iyy, pint.Quantity) and not np.isnan(section.Iyy):
                section.Iyy = section.Iyy.to(internal_inertia_unit)

            log(f"Section {section.uid}: Area={section.Area}, Ixx={section.Ixx}, Iyy={section.Iyy}")

        # Add section to dictionary
        sections_dict[section.uid] = section

    log(f"Loaded {len(sections_dict)} sections")
    return sections_dict