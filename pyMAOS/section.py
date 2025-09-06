import numpy as np
from pyMAOS.unit_aware import UnitAwareMixin
from pyMAOS.units_mod import unit_manager
from typing import Union
import pint
from pint import Quantity
from math import isnan

class Section(UnitAwareMixin):
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
        from pyMAOS.units_mod import unit_manager

        # Get values from state
        uid = state.get('uid')
        area = state.get('area', "10 in^2")
        ixx_value = state.get('ixx', np.nan)
        iyy_value = state.get('iyy', np.nan)

        # Convert to Quantity objects or leave as NaN
        area_qty = unit_manager.ureg(area)
        ixx_qty = unit_manager.ureg(ixx_value) if not isinstance(ixx_value, float) or not np.isnan(ixx_value) else np.nan
        iyy_qty = unit_manager.ureg(iyy_value) if not isinstance(iyy_value, float) or not np.isnan(iyy_value) else np.nan

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

        # If it's already a Quantity, convert to appropriate units
        import pint
        if isinstance(value, pint.Quantity):
            if unit_type == 'area':
                return value.to('m^2')
            elif unit_type == 'moment_of_inertia':
                return value.to('m^4')
            return value

        # If it's a number, assume SI units and convert to Quantity
        if isinstance(value, (int, float)):
            if unit_type == 'area':
                return value * unit_manager.ureg.m ** 2
            elif unit_type == 'moment_of_inertia':
                return value * unit_manager.ureg.m ** 4
            return value * unit_manager.ureg.dimensionless

        # If it's a string, parse units
        if isinstance(value, str):
            try:
                from pyMAOS.units_mod import parse_value_with_units
                parsed_value = parse_value_with_units(value)

                if isinstance(parsed_value, pint.Quantity):
                    if unit_type == 'area':
                        return parsed_value.to('m^2')
                    elif unit_type == 'moment_of_inertia':
                        return parsed_value.to('m^4')
                    return parsed_value
                else:
                    # No units in string, create with default units
                    if unit_type == 'area':
                        return parsed_value * unit_manager.ureg.m ** 2
                    elif unit_type == 'moment_of_inertia':
                        return parsed_value * unit_manager.ureg.m ** 4
                    return parsed_value * unit_manager.ureg.dimensionless

            except Exception as e:
                print(f"Warning: Could not parse section value '{value}': {e}")
                try:
                    num_value = float(value)
                    if unit_type == 'area':
                        return num_value * unit_manager.ureg.m ** 2
                    elif unit_type == 'moment_of_inertia':
                        return num_value * unit_manager.ureg.m ** 4
                    return num_value * unit_manager.ureg.dimensionless
                except:
                    raise ValueError(f"Could not parse section value: {value}")

        raise ValueError(f"Unsupported section value type: {type(value)}")

    def __str__(self):
        """Return string representation of the section properties"""
        units = unit_manager.get_current_units()
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