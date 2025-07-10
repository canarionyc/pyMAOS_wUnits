# -*- coding: utf-8 -*-

class LinearElasticMaterial:
    def __init__(self, density=0.00028, E=29000):

        self.density = density  # Material Density for Self Weight
        self.E = E  # Young's Modulus -- Modulus of Elasticity

    def stress(self, strain):

        return self.E * strain
