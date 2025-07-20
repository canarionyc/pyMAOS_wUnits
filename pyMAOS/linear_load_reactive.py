from rx.subject.subject import Subject
import rx
from rx import operators as ops
import numpy as np

class LinearLoadReactive:
    """Reactive implementation of linear load processing"""
    
    def __init__(self):
        # Create subjects for load parameters
        self.w1_subject = Subject()
        self.w2_subject = Subject()
        self.a_subject = Subject()
        self.b_subject = Subject()
        self.L_subject = Subject()
        
        # Define output streams
        self.total_load = rx.combine_latest(
            self.w1_subject, 
            self.w2_subject,
            self.a_subject,
            self.b_subject
        ).pipe(
            ops.map(lambda params: self.calculate_total_load(*params)),
            ops.share()
        )
        
        self.load_centroid = rx.combine_latest(
            self.w1_subject,
            self.w2_subject,
            self.a_subject,
            self.b_subject
        ).pipe(
            ops.map(lambda params: self.calculate_load_centroid(*params)),
            ops.share()
        )
        
        # Calculate reactions reactively
        self.reactions = rx.combine_latest(
            self.total_load,
            self.load_centroid,
            self.a_subject,
            self.L_subject
        ).pipe(
            ops.map(lambda params: self.calculate_reactions(*params)),
            ops.share()
        )
        
        # Calculate integration constants
        self.constants = rx.combine_latest(
            self.w1_subject, 
            self.w2_subject,
            self.a_subject, 
            self.b_subject,
            self.L_subject
        ).pipe(
            ops.map(lambda params: self.calculate_constants(*params)),
            ops.share()
        )
    
    def calculate_total_load(self, w1, w2, a, b):
        """Calculate total load (trapezoidal area)"""
        c = b - a
        return 0.5 * c * (w1 + w2)
    
    def calculate_load_centroid(self, w1, w2, a, b):
        """Calculate load centroid position relative to point a"""
        c = b - a
        return c * (w1 + 2*w2) / (3*(w1 + w2))
    
    def calculate_reactions(self, W, c_bar, a, L):
        """Calculate simple support reactions"""
        R_j = -W * (a + c_bar) / L
        R_i = -W - R_j
        return (R_i, R_j)
    
    def calculate_constants(self, w1, w2, a, b, L):
        """Calculate integration constants c1-c12"""
        # Implement the formulas from linear_load_formulas.md
        # Example for c1:
        c1 = ((2*b**2 + (-a-3*L)*b - a**2 + 3*L*a)*w2 + 
              (b**2 + (a-3*L)*b - 2*a**2 + 3*L*a)*w1) / (6*L)
              
        # Continue with other constants...
        c2 = ((2*b**3 + (-3*a-3*L)*b**2 + 6*L*a*b + a**3)*w2 + 
              (b**3 - 3*L*b**2 - 3*a**2*b + 2*a**3)*w1) / (6*L*(b-a))
        
        c3 = ((2*b**2 - a*b - a**2)*w2 + (b**2 + a*b - 2*a**2)*w1) / (6*L)
        
        # Return all constants
        return {"c1": c1, "c2": c2, "c3": c3}
    
    def set_parameters(self, w1, w2, a, b, L):
        """Set all load parameters at once"""
        self.w1_subject.on_next(w1)
        self.w2_subject.on_next(w2)
        self.a_subject.on_next(a)
        self.b_subject.on_next(b)
        self.L_subject.on_next(L)