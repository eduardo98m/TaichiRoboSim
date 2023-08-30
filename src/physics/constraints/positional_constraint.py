import taichi as ti
import taichi.math as tm

import quaternion
from bodies import RigidBody

from .constraint import Constraint, ConstraintResponse


EPSILON = 1e-50

@ti.func
def positional_constraint_lagrange_multiplier_update(
    c             : ti.types.f32,
    w_1           : ti.types.f32,
    w_2           : ti.types.f32,
    lagrange_mult : ti.types.f32,
    h             : ti.types.f32,
    alpha         : ti.types.f32
): 
    
    """
        Arguments:
        ----------
        `c` : ti.types.f32
            -> Constraint value
        w_1 : ti.types.f32
            -> Generalized inverse mass of object 1
        w_2 : ti.types.f32
            -> Generalized inverse mass of object 2
        lambda_ : ti.types.f32
            -> Lagrange multiplier
        h : ti.types.f32
            -> Time step
        alpha: ti.types.f32
            -> Contraint compliance 
    """
    alpha_p = alpha / h**2
    return  (-c - alpha_p * lagrange_mult )/ (w_1 + w_2 + alpha_p)



@ti.func
def positional_contraint_generalized_inverse_mass(
                                                  m     : ti.types.f32,
                                                  I_inv : ti.types.matrix(3,3, ti.types.f32),
                                                  r     : ti.types.vector(3, ti.types.f32),
                                                  n     : ti.types.vector(3, ti.types.f32)
                                                  ):
    """
        Arguments:
        ----------
        m : ti.types.f32
            -> Mass
        I_inv : ti.types.matrix(3,3, ti.types.f32)
            -> Inverse inertia matrix
        r : ti.types.vector(3, ti.types.f32)
            -> Vector relative to the center of mass of the object, where the constraint is applied.
        n : ti.types.vector(3, ti.types.f32)
            -> Normal vector of the constraint
    
    """
    return 1 / m + tm.dot(tm.cross(r, n), I_inv @ tm.cross(r, n))

@ti.func
def compute_positional_constraint(
    body_1     : RigidBody,
    body_2     : RigidBody,
    constraint : Constraint,
    h          : ti.types.f32,
) -> ConstraintResponse:
    """
        Arguments:
        ----------
        `obj_1` : RigidBody
            -> Object 1
        `obj_2` : RigidBody
            -> Object 2  
        `constraint` : Constraint
            -> Constraint object
        `h` : ti.types.f32
            -> Time step (Substep)     
    """
    n = constraint.direction
    c = constraint.magnitude

    r_1_lc = constraint.r_1
    r_2_lc = constraint.r_2

    lagrange_mult = constraint.lagrange_mult

    compliance = constraint.compliance

    delta_lagrange_mult = 0.0
    
    w_1 = 0.0
    w_2 = 0.0
    r_1 = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    r_2 = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    if c >= EPSILON:
        # We need to rotate the r vectors to the world frame
        # And Calculate the generalized inverse mass
        
        r_1 = quaternion.rotate_vector(body_1.orientation, r_1_lc)
        w_1 = body_1.compute_positional_generalized_inverse_mass(r_1, n)
        
        
        r_2 = quaternion.rotate_vector(body_2.orientation, r_2_lc)
        w_2 = body_2.compute_positional_generalized_inverse_mass(r_2, n)

        # Calculate the Lagrange multiplier update
        delta_lagrange_mult = positional_constraint_lagrange_multiplier_update(c, w_1, w_2, lagrange_mult, h, compliance)

        impulse = delta_lagrange_mult * n 

        # Compute the new position and orientation 

        body_1.apply_position_correction(impulse, r_1)
        body_2.apply_position_correction(-impulse, r_2)
        
    
    force = (lagrange_mult + delta_lagrange_mult) * n / h**2
    return ConstraintResponse(
        force               = force,
        new_lagrange_mult   = lagrange_mult + delta_lagrange_mult,
        new_position_1      = body_1.position,
        new_orientation_1   = body_1.orientation,
        new_position_2      = body_2.position,
        new_orientation_2   = body_2.orientation
    )

