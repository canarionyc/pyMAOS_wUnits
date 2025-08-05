

class UnitAwareMixin:
    """
    Mixin class providing unit conversion capabilities for model components
    
    Classes inheriting from this mixin gain consistent unit handling
    capabilities that integrate with the central unit manager.
    """
    
    def __init__(self):
        # Register for unit system updates
        unit_manager.register_for_unit_updates(self._update_units)
        
        # Store internal values in SI
        self._internal_values = {}
        self._display_values = {}
        
    def _update_units(self, new_unit_system):
        """Called when the global unit system changes"""
        # Recalculate display values based on internal SI values
        for key, value in self._internal_values.items():
            unit_type = self._get_unit_type_for_attribute(key)
            if unit_type:
                si_unit = self._get_si_unit_for_type(unit_type)
                display_unit = new_unit_system.get(unit_type, si_unit)
                self._display_values[key] = unit_manager.convert_value(
                    value, si_unit, display_unit)
    
    def _get_unit_type_for_attribute(self, attr_name):
        """Map attribute names to unit types"""
        # Override in subclasses with specific mapping
        unit_mappings = {
            'length': 'length',
            'area': 'area',
            'Area': 'area',
            'ixx': 'moment_of_inertia',
            'Ixx': 'moment_of_inertia',
            'iyy': 'moment_of_inertia',
            'Iyy': 'moment_of_inertia',
            'E': 'pressure',
            'density': 'density',
            'force': 'force',
            'moment': 'moment'
        }
        return unit_mappings.get(attr_name)
    
    def _get_si_unit_for_type(self, unit_type):
        """Get the SI unit for a given unit type"""
        si_units = {
            'length': 'm',
            'area': 'm^2',
            'moment_of_inertia': 'm^4',
            'pressure': 'Pa',
            'density': 'kg/m^3',
            'force': 'N',
            'moment': 'N*m',
            'distributed_load': 'N/m'
        }
        return si_units.get(unit_type, '')
        
    def set_value_with_units(self, attr_name, value, unit=None):
        """
        Set a value with appropriate unit conversion
        
        Stores the internal value in SI units and calculates the display value
        """
        unit_type = self._get_unit_type_for_attribute(attr_name)
        if not unit_type:
            # No unit handling for this attribute
            setattr(self, attr_name, value)
            return
            
        si_unit = self._get_si_unit_for_type(unit_type)
        
        # If unit is specified, convert to SI
        if unit and unit != si_unit:
            internal_value = unit_manager.convert_value(value, unit, si_unit)
        else:
            internal_value = value
            
        # Store internal (SI) value
        self._internal_values[attr_name] = internal_value
        
        # Calculate display value
        current_units = unit_manager.get_current_units()
        display_unit = current_units.get(unit_type, si_unit)
        display_value = unit_manager.convert_value(internal_value, si_unit, display_unit)
        self._display_values[attr_name] = display_value
        
        # Set the actual attribute to the internal value (SI units)
        setattr(self, attr_name, internal_value)
        
    def get_value_in_units(self, attr_name, unit=None):
        """
        Get a value in the specified units
        
        If no unit is specified, returns the value in display units
        """
        internal_value = self._internal_values.get(attr_name)
        if internal_value is None:
            internal_value = getattr(self, attr_name, None)
            
        if internal_value is None:
            return None
            
        unit_type = self._get_unit_type_for_attribute(attr_name)
        if not unit_type:
            return internal_value
            
        si_unit = self._get_si_unit_for_type(unit_type)
        
        if unit:
            return unit_manager.convert_value(internal_value, si_unit, unit)
        else:
            # Use current display unit
            current_units = unit_manager.get_current_units()
            display_unit = current_units.get(unit_type, si_unit)
            return unit_manager.convert_value(internal_value, si_unit, display_unit)
