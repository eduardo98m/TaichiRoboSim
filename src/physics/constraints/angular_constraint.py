import taichi as ti
import taichi.math as tm

import quaternion
from bodies import RigidBody

from .constraint import Constraint, ConstraintResponse

EPSILON = 1e-8
@ti.func
def angular_constraint_lagrange_multiplier_update(
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
        `c`   : ti.types.f32
            -> Constraint value
        `w_1` : ti.types.f32
            -> Generalized inverse mass of object 1
        `w_2` : ti.types.f32
            -> Generalized inverse mass of object 2
        `lagrange_mult` : ti.types.f32
            -> Lagrange multiplier
        `h`   : ti.types.f32
            -> Time step
        `alpha`: ti.types.f32
            -> Contraint compliance 
    """
    alpha_p = alpha / h**2
    return  -c - alpha_p * lagrange_mult / (w_1 + w_2 + alpha_p)


@ti.func
def angular_contraint_generalized_inverse_mass( 
                                                I_inv : ti.types.matrix(3,3, ti.types.f32),
                                                n     : ti.types.vector(3, ti.types.f32)
                                               ):
    """
        Arguments:
        ----------
        I_inv : ti.types.matrix(3,3, ti.types.f32)
            -> Inverse inertia matrix
        n : ti.types.vector(3, ti.types.f32)
            -> Normal vector of the constraint
    """
    return n @ I_inv @ n

@ti.func
def compute_angular_constraint(
    body_1         : RigidBody,
    body_2         : RigidBody,
    constraint     : Constraint,
    h              : ti.types.f32 
) -> ConstraintResponse:
    """
    
        Arguments:
        ----------
        `body_1` : RigidBody
            -> Object 1
        `body_2` : RigidBody
            -> Object 2
        `h` : ti.types.f32
            -> Time step (Substep)
        `direction` : ti.types.vector(3, ti.types.f32)
            -> Direction of the constraint
        `magnitude` : ti.types.f32
            -> Magnitude of the constraint
        `lagrange_mult` : ti.types.f32
            -> Lagrange multiplier of the constraint    
    """
    
    n     = constraint.direction
    theta = constraint.magnitude
    lagrange_mult  = constraint.lagrange_mult
    compliance    = constraint.compliance

    delta_lagrange_mult = 0.0
    
    body_1.update_inertia()
    body_2.update_inertia()
    
    if theta >= EPSILON:
        # Calculate the generalized inverse mass
        w_1 = angular_contraint_generalized_inverse_mass(body_1.dynamic_inv_interia, n)
        w_2 = angular_contraint_generalized_inverse_mass(body_2.dynamic_inv_interia, n)
        # Calculate the Lagrange multiplier update
        delta_lagrange_mult =  angular_constraint_lagrange_multiplier_update(theta, w_1, w_2, lagrange_mult, h, compliance)

        
    
    # claculate the angular impulse
    impulse = delta_lagrange_mult * n


    new_orientation_1 = body_1.orientation
    new_orientation_2 = body_2.orientation

    # update the orientation
    if not body_1.fixed:
        new_orientation_1 = quaternion.rotate_by_axis(q = body_1.orientation, 
                                                     axis = body_1.dynamic_inv_interia @ impulse
                                                     )
        
    
    if not body_2.fixed:
        new_orientation_2 = quaternion.rotate_by_axis(q = body_2.orientation, 
                                                     axis = - body_2.dynamic_inv_interia @ impulse
                                                     )
        
    torque = impulse/ h**2

    return ConstraintResponse(
        torque              = torque,
        new_lagrange_mult   = lagrange_mult + delta_lagrange_mult,
        new_orientation_1   = new_orientation_1,
        new_orientation_2   = new_orientation_2
    )

@ti.func
def angular_constraint_velocity_update(body_1:RigidBody, 
                                       body_2 :RigidBody, 
                                       correction : ti.types.vector(3, ti.f32), 
                                       h : ti.f32):
    response = ConstraintResponse(
        torque             = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
        new_orientation_1  = body_1.orientation,
        new_orientation_2  = body_2.orientation
    )

    if tm.length(correction) > EPSILON:

        # Calculate the impulse:
        n  = tm.normalize(correction)

        body_1.update_inertia()
        body_2.update_inertia()
        # Calculate the generalized inverse mass
        w_1 = angular_contraint_generalized_inverse_mass(body_1.dynamic_inv_interia, n)
        w_2 = angular_contraint_generalized_inverse_mass(body_2.dynamic_inv_interia, n)

        impulse = correction / (w_1 + w_2)

        new_orientation_1 = body_1.orientation
        new_orientation_2 = body_2.orientation

        # update the orientation
        if not body_1.fixed:
            new_orientation_1 = quaternion.rotate_by_axis(q = body_1.orientation, 
                                                        axis = body_1.dynamic_inv_interia @ impulse
                                                        )
            
        
        if not body_2.fixed:
            new_orientation_2 = quaternion.rotate_by_axis(q = body_2.orientation, 
                                                        axis = - body_2.dynamic_inv_interia @ impulse
                                                        )
        
        torque = impulse/ h**2

        response = ConstraintResponse(
            torque             = torque,
            new_orientation_1  = new_orientation_1,
            new_orientation_2  = new_orientation_2
            )

    return response
