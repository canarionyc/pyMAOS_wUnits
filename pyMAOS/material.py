from pyMAOS.unit_aware import UnitAwareMixin
from pyMAOS.units_mod import unit_manager
from typing import Union
import pint
from pint import Quantity
class LinearElasticMaterial(UnitAwareMixin):
    def __init__(self, uid: int,
                 density: Union[float, str, Quantity] = 7861.092937697687,
                 E: Union[float, str,Quantity] = 199947961501.8826,
                 nu: float = 0.3):
        super().__init__()
        self.uid = uid  # Unique Material Identifier
        self.nu = nu  # Poisson's ratio (dimensionless)

        import pint
        if isinstance(E, pint.Quantity):
            # If it's a pint Quantity, convert to SI units
            self.E=E.to('Pa')  # .magnitude
        else:
            # If it's a float or string, parse it
            self.E = self._parse_value_with_units(E, 'pressure')
            if isinstance(self.E, pint.Quantity):
                self.E = self.E.to('Pa')


        if isinstance(density, pint.Quantity):
            self.density=density.to('kg/m^3')  # .magnitude
        else:
            # If it's a float or string, parse it
            density = self._parse_value_with_units(density, 'density')
            if isinstance(density, pint.Quantity):
                density = density.to('kg/m^3')

        print(
            f"LinearElasticMaterial uid {self.uid} initialized with density={self.density:.4g} kg/m^3, E={self.E:.3g} Pa, nu={self.nu}")

    def __setstate__(self, state):

        from pyMAOS.units_mod import ureg
        # Initialize the object properly
        self.__init__(state.get('uid'),
                      ureg(state.get('density', "0.284 lb/in^3")).to('kg/m^3'),
                      ureg(state.get('E', "29000.0 ksi")).to('Pa'),
                      state.get('nu', 0.3))

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

        units = unit_manager.get_current_units()
        e_display = self.E.to(units['pressure'])
        density_display = self.density.to(units['density'])
        # pressure_unit = units.get('pressure', 'Pa')
        # density_unit = units.get('density', 'kg/m^3')
        return f"LinearElasticMaterial(uid={self.uid}, density={density_display:.4f}, E={e_display}, nu={self.nu}) in {units.get('name', 'default')} units"


    def __repr__(self):
        """Return developer representation of the material"""
        return f"LinearElasticMaterial(uid={self.uid}, density={self.density:.4f}, E={self.E:.2f}, nu={self.nu})"
