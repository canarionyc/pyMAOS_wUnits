# -*- coding: utf-8 -*-

class LinearElasticMaterial:
    def __init__(self, density=0.00028, E=29000, nu=0.3):
        self.density = density  # Material Density for Self Weight
        self.E = E  # Young's Modulus -- Modulus of Elasticity
        self.nu = nu

    def stress(self, strain):
        return self.E * strain
        
    def __str__(self):
        """Return string representation of the material properties"""
        return f"LinearElasticMaterial(density={self.density}, E={self.E}, nu={self.nu})"
    
    def __repr__(self):
        """Return developer representation of the material"""
        return self.__str__()
