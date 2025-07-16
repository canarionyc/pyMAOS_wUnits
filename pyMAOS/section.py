import numpy as np

class Section:
    def __init__(self, uid, Area=10.0, Ixx=np.nan, Iyy=np.nan):
        self.uid = uid
        self.Area = Area  # Cross-Section Area
        self.Ixx = Ixx  # Cross-Section centroidal Moment of Inertia, strong
        self.Iyy = Iyy  # Cross-Section centroidal Moment of inertia, weak
        
    def __str__(self):
        """Return string representation of the section properties"""
        return f"Section(uid={self.uid}, Area={self.Area}, Ixx={self.Ixx}, Iyy={self.Iyy})"

    def __repr__(self):
        """Return developer representation of the section"""
        return self.__str__()
