import numpy as np
from pyMAOS.unit_aware import UnitAwareMixin
from pyMAOS.units_mod import unit_manager
from typing import Union

class Section(UnitAwareMixin):
    def __init__(self, uid: int, Area: Union[float, str] = 10.0, Ixx: Union[float, str] = np.nan, Iyy: Union[float, str] = np.nan):
        super().__init__()
        self.uid = uid
        
        # Parse and store values with proper unit handling
        self.Area = self._parse_value_with_units(Area, 'area')
        self.Ixx = self._parse_value_with_units(Ixx, 'moment_of_inertia') # if not self._is_nan_value(Ixx) else np.nan
        self.Iyy = self._parse_value_with_units(Iyy, 'moment_of_inertia') # if not self._is_nan_value(Iyy) else np.nan
        
        # Store using the UnitAwareMixin methods for proper unit management
        self.set_value_with_units('Area', self.Area)
        self.set_value_with_units('Ixx', self.Ixx)
        self.set_value_with_units('Iyy', self.Iyy)

    def _is_nan_value(self, value: Union[float, str]) -> bool:
        """Check if a value is NaN, handling both numeric and string inputs"""
        try:
            if isinstance(value, str):
                # Try to convert string to float first
                numeric_value = float(value)
                return np.isnan(numeric_value)
            else:
                return np.isnan(float(value))
        except (ValueError, TypeError):
            return False

    def _parse_value_with_units(self, value: Union[float, str], unit_type: str) -> float:
        """
        Parse a value that may be a float or string with units.
        
        Parameters
        ----------
        value : float or str
            Value, either as a number or string with units (e.g., "11.8in^2", "518in^4")
        unit_type : str
            Type of unit ('area', 'moment_of_inertia')
            
        Returns
        -------
        float
            Value in SI units
        """
        # First check if value is NaN
        if isinstance(value, (int, float)) and np.isnan(float(value)):
            return np.nan
        
        # If it's already a number, return it as-is (assume SI units)
        if isinstance(value, (int, float)):
            return float(value)
        
        # If it's a string, try to parse units
        if isinstance(value, str):
            try:
                # Import the parse function from units_mod
                from pyMAOS.units_mod import parse_value_with_units
                import pint
                
                # Parse the value string
                parsed_value = parse_value_with_units(value)
                
                # If it has units, convert to SI units
                if isinstance(parsed_value, pint.Quantity):
                    try:
                        # Convert based on unit type
                        if unit_type == 'area':
                            si_value = parsed_value.to('m^2').magnitude
                        elif unit_type == 'moment_of_inertia':
                            si_value = parsed_value.to('m^4').magnitude
                        else:
                            # Fallback - just use the magnitude
                            si_value = parsed_value.magnitude
                        return float(si_value)
                    except Exception as e:
                        print(f"Warning: Could not convert '{value}' to SI units for {unit_type}: {e}")
                        # Fall back to magnitude if conversion fails
                        return float(parsed_value.magnitude)
                else:
                    # No units, just return the numeric value
                    return float(parsed_value)
                    
            except Exception as e:
                print(f"Warning: Could not parse section value '{value}': {e}")
                # Try to convert directly to float as fallback
                try:
                    return float(value)
                except Exception:
                    raise ValueError(f"Could not parse section value: {value}")
        
        # If we get here, something unexpected happened
        raise ValueError(f"Unsupported section value type: {type(value)}")

    def __str__(self):
        area_display = self.get_value_in_units('Area')
        ixx_display = self.get_value_in_units('Ixx')
        iyy_display = self.get_value_in_units('Iyy')
        units = unit_manager.get_current_units()
        area_unit = units.get('area', 'm^2')
        ixx_unit = units.get('moment_of_inertia', 'm^4')
        return f"Section {self.uid}: A={area_display} {area_unit}, Ixx={ixx_display} {ixx_unit}, Iyy={iyy_display} {ixx_unit}"

    def __repr__(self):
        """Return developer representation of the section"""
        return self.__str__()
