from pyMAOS.unit_aware import UnitAwareMixin
from pyMAOS.units_mod import unit_manager
from typing import Union


class LinearElasticMaterial(UnitAwareMixin):
    def __init__(self, uid: int, density: Union[float, str] = 0.00028, E: Union[float, str] = 29000.0, nu: float = 0.3):
        super().__init__()
        self.uid = uid  # Unique Material Identifier
        self.nu = nu   # Poisson's ratio (dimensionless)
        
        # Parse and store values with proper unit handling
        self.density = self._parse_value_with_units(density, 'density')
        self.E = self._parse_value_with_units(E, 'pressure')  # Young's modulus is a pressure unit
        
        # Store using the UnitAwareMixin methods for proper unit management
        self.set_value_with_units('density', self.density)
        self.set_value_with_units('E', self.E)
        
        print(f"LinearElasticMaterial uid {self.uid} initialized with density={self.density} kg/m^3, E={self.E} Pa, nu={self.nu}")

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
                from pyMAOS.units_mod import parse_value_with_units
                import pint
                
                # Parse the value string
                parsed_value = parse_value_with_units(value)
                
                # If it has units, convert to SI units
                if isinstance(parsed_value, pint.Quantity):
                    try:
                        # Convert based on unit type
                        if unit_type == 'pressure':
                            # Young's modulus - convert to Pascals
                            si_value = parsed_value.to('Pa').magnitude
                        elif unit_type == 'density':
                            # Density - convert to kg/m³
                            si_value = parsed_value.to('kg/m^3').magnitude
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
                print(f"Warning: Could not parse material value '{value}': {e}")
                # Try to convert directly to float as fallback
                try:
                    return float(value)
                except Exception:
                    raise ValueError(f"Could not parse material value: {value}")
        
        # If we get here, something unexpected happened
        raise ValueError(f"Unsupported material value type: {type(value)}")

    def stress(self, strain):
        return self.E * strain
        
    def __str__(self):
        """Return string representation of the material properties"""
        e_display = self.get_value_in_units('E')
        density_display = self.get_value_in_units('density')
        units = unit_manager.get_current_units()
        pressure_unit = units.get('pressure', 'Pa')
        density_unit = units.get('density', 'kg/m^3')
        return f"LinearElasticMaterial(uid={self.uid}, density={density_display} {density_unit}, E={e_display} {pressure_unit}, nu={self.nu})"

    def __repr__(self):
        """Return developer representation of the material"""
        return self.__str__()
