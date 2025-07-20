import numpy as np
from pyMAOS.unit_aware import UnitAwareMixin

class Section(UnitAwareMixin):
    def __init__(self, uid, Area=10.0, Ixx=np.nan, Iyy=np.nan):
        super().__init__()
        self.uid = uid
        # Store values with proper unit handling
        self.set_value_with_units('Area', Area)
        self.set_value_with_units('Ixx', Ixx)
        self.set_value_with_units('Iyy', Iyy)

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
