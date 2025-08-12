import numpy as np
import pyMAOS
from pyMAOS.unit_aware import UnitAwareMixin
from quantity_utils import convert_registry

from typing import Union
import pint
from pint import Quantity
from math import isnan

class Section():
    def __init__(self, uid: int,
                 Area: Union[str, Quantity],
                 Ixx: Union[str, Quantity] = "1.0 in^4",
                 Iyy: Union[str, Quantity] = "1.0 in^4",
                 name: str = None):
        """
        Initialize a section with proper unit validation.

        Parameters
        ----------
        uid : int
            Unique Section Identifier
        Area : Union[str, Quantity]
            Cross-sectional area with units (e.g. "10 in^2")
        Ixx : Union[str, Quantity], optional
            Moment of inertia about x-axis with units (e.g. "100 in^4")
        Iyy : Union[str, Quantity], optional
            Moment of inertia about y-axis with units (e.g. "50 in^4")
        name : str, optional
            Name identifier for the section
        """
        super().__init__()
        self.uid = uid
        self.name = name if name else f"Section_{uid}"

        # Get internal units from unit manager
        internal_area_unit = pyMAOS.unit_manager.INTERNAL_AREA_UNIT
        internal_inertia_unit = pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT
        system_name = pyMAOS.unit_manager.system_name

        print(f"Creating section {uid} using internal unit system: {system_name}")
        print(f"Internal area unit: {internal_area_unit}")
        print(f"Internal inertia unit: {internal_inertia_unit}")

        # Handle Area - must be a string with units or a Quantity
        if isinstance(Area, (int, float)):
            raise TypeError("Area must be provided as a string with units or a Quantity object")

        # Convert string to Quantity if needed
        if not isinstance(Area, pint.Quantity):
            Area = pyMAOS.unit_manager.ureg(Area)

        # Validate dimensions for Area
        if not Area.check({'[length]': 2}):
            raise ValueError(f"Area has incorrect dimensions: {Area.dimensionality}. "
                            f"Expected dimensions: [length]^2")

        # Store validated value in internal units
        self.Area = Area.to(pyMAOS.unit_manager.INTERNAL_AREA_UNIT)

        # Process Ixx - must be a string with units or a Quantity
        if isinstance(Ixx, (int, float)):
            raise ValueError("Ixx must be provided as a string with units or a Quantity object")

        # Convert string to Quantity if needed
        if not isinstance(Ixx, pint.Quantity):
            try:
                Ixx = pyMAOS.unit_manager.ureg(Ixx)
            except Exception as e:
                raise ValueError(f"Invalid Ixx value: {Ixx}. Error: {e}")

        # Validate dimensions for Ixx
        if not Ixx.check({'[length]': 4}):
            raise ValueError(f"Ixx has incorrect dimensions: {Ixx.dimensionality}. "
                           f"Expected dimensions: [length]^4")

        # Convert to internal units
        Ixx_internal = Ixx.to(internal_inertia_unit)

        # Check for zero or negative values
        if Ixx_internal.magnitude <= 0:
            raise ValueError(f"Ixx must be greater than zero. Got: {Ixx}")

        self.Ixx = Ixx_internal
        print(f"Processed Ixx = {self.Ixx}")

        # Process Iyy - must be a string with units or a Quantity
        if isinstance(Iyy, (int, float)):
            raise ValueError("Iyy must be provided as a string with units or a Quantity object")

        # Convert string to Quantity if needed
        if not isinstance(Iyy, pint.Quantity):
            try:
                Iyy = pyMAOS.unit_manager.ureg(Iyy)
            except Exception as e:
                raise ValueError(f"Invalid Iyy value: {Iyy}. Error: {e}")

        # Validate dimensions for Iyy
        if not Iyy.check({'[length]': 4}):
            raise ValueError(f"Iyy has incorrect dimensions: {Iyy.dimensionality}. "
                           f"Expected dimensions: [length]^4")

        # Convert to internal units
        Iyy_internal = Iyy.to(internal_inertia_unit)

        # Check for zero or negative values
        if Iyy_internal.magnitude <= 0:
            raise ValueError(f"Iyy must be greater than zero. Got: {Iyy}")

        self.Iyy = Iyy_internal
        print(f"Processed Iyy = {self.Iyy}")

    def __setstate__(self, state):
        """
        Restore the object state during unpickling.

        Parameters
        ----------
        state : dict
            Dictionary containing the object state
        """
        # Handle Area value properly based on its type
        area_value = state.get('Area', "1.0 in^2")
        if isinstance(area_value, (int, float)):
            pyMAOS.warning(f"assuming Area {area_value} is in {pyMAOS.unit_manager.INTERNAL_AREA_UNIT}")
            area_quantity = pyMAOS.unit_manager.ureg.Quantity(area_value, pyMAOS.unit_manager.INTERNAL_AREA_UNIT)
        elif isinstance(area_value, pint.Quantity):
            # If it's already a Quantity object, use it directly
            area_quantity = convert_registry(area_value, pyMAOS.unit_manager.ureg)
        else:
            # If it's a string, parse it with ureg
            area_quantity = pyMAOS.unit_manager.ureg(area_value)

        # Handle Ixx value properly based on its type
        ixx_value = state.get('Ixx', "1.0 in^4")
        if isinstance(ixx_value, (int, float)):
            if np.isnan(ixx_value) or ixx_value <= 0:
                ixx_value = "1.0 in^4"  # Use a non-zero default with units
                pyMAOS.warning(f"Invalid Ixx value detected, using default: {ixx_value}")
            else:
                pyMAOS.warning(f"assuming Ixx {ixx_value} is in {pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT}")
                ixx_quantity = pyMAOS.unit_manager.ureg.Quantity(ixx_value, pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT)
        elif isinstance(ixx_value, pint.Quantity):
            # If it's already a Quantity object, use it directly
            ixx_quantity = convert_registry(ixx_value, pyMAOS.unit_manager.ureg)
        else:
            # If it's a string, parse it with ureg
            ixx_quantity = pyMAOS.unit_manager.ureg(ixx_value)

        # Handle Iyy value properly based on its type
        iyy_value = state.get('Iyy', "1.0 in^4")
        if isinstance(iyy_value, (int, float)):
            if np.isnan(iyy_value) or iyy_value <= 0:
                iyy_value = "1.0 in^4"  # Use a non-zero default with units
                pyMAOS.warning(f"Invalid Iyy value detected, using default: {iyy_value}")
            else:
                pyMAOS.warning(f"assuming Iyy {iyy_value} is in {pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT}")
                iyy_quantity = pyMAOS.unit_manager.ureg.Quantity(iyy_value, pyMAOS.unit_manager.INTERNAL_MOMENT_OF_INERTIA_UNIT)
        elif isinstance(iyy_value, pint.Quantity):
            # If it's already a Quantity object, use it directly
            iyy_quantity = convert_registry(iyy_value, pyMAOS.unit_manager.ureg)
        else:
            # If it's a string, parse it with ureg
            iyy_quantity = pyMAOS.unit_manager.ureg(iyy_value)

        # Get name if it exists
        name = state.get('name', None)

        # Pass the values directly to __init__ for validation and processing
        self.__init__(
            uid=state.get('uid', 0),
            Area=area_quantity,
            Ixx=ixx_quantity,
            Iyy=iyy_quantity,
            name=name
        )

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

        return f"Section {self.name} (ID: {self.uid}): A={area_display:.2f}, Ixx={ixx_display:.2f}, Iyy={iyy_display}"

    def __repr__(self):
        """Return developer representation of the section"""
        return f"Section(uid={self.uid}, name='{self.name}', Area={self.Area}, Ixx={self.Ixx}, Iyy={self.Iyy})"

def get_sections_from_yaml(sections_yml, logger=None):
    """
    Loads sections from a YAML file and returns them as a dictionary.

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
    def log(message):
        if logger:
            pyMAOS.info(message)
        else:
            print(message)

    pyMAOS.info(f"Loading sections from: {sections_yml}")

    try:
        import yaml
        with open(sections_yml, 'r') as file:
            sections_list = yaml.unsafe_load(file)

        # Convert list to dictionary with uid as key
        sections_dict = {section.uid: section for section in sections_list}

        pyMAOS.info(f"Loaded {len(sections_dict)} sections")
        return sections_dict
    except Exception as e:
        pyMAOS.info(f"Error loading sections from {sections_yml}: {e}")
        return {}
