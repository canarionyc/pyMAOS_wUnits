# -*- coding: utf-8 -*-

class Section:
    def __init__(self, Area=10, Ixx=10, Iyy=10):
        self.Area = Area  # Cross-Section Area
        self.Ixx = Ixx  # Cross-Section centroidal Moment of Inertia, strong
        self.Iyy = Iyy  # Cross-Section centroidal Moment of inertia, weak
        
    def __str__(self):
        """Return string representation of the section properties"""
        return f"Section(Area={self.Area}, Ixx={self.Ixx}, Iyy={self.Iyy})"
    
    def __repr__(self):
        """Return developer representation of the section"""
        return self.__str__()
