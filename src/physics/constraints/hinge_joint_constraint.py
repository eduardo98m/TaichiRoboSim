from base_constraints import *
import taichi as ti
import taichi.math as tm
from quaternion import quaternion

@ti.dataclass
class HingeJointConstraint:
    compliance         : ti.types.f32
    r1_lc              : ti.types.vector(3, ti.types.f32)
    r2_lc              : ti.types.vector(3, ti.types.f32)
    e1_aligned_axis    : ti.types.vector(3, ti.types.f32)
    e2_aligned_axis    : ti.types.vector(3, ti.types.f32)
    e1_limit_axis      : ti.types.vector(3, ti.types.f32)
    e2_limit_axis      : ti.types.vector(3, ti.types.f32)
    limited            : bool
    lower_limit        : ti.types.f32
    upper_limit        : ti.types.f32
    lambda_aligned     : ti.types.f32
    lambda_positional  : ti.types.f32
    lambda_limit       : ti.types.f32

@ti.dataclass
class HingeJointConstraintResponse:
    force                 : ti.types.vector(3, ti.types.f32) 
    torque                : ti.types.vector(3, ti.types.f32)
    new_lambda_aligned    : ti.types.f32
    new_lambda_positional : ti.types.f32
    new_lambda_limit      : ti.types.f32
    new_position_1        : ti.types.vector(3, ti.types.f32)
    new_orientation_1     : ti.types.vector(4, ti.types.f32)
    new_position_2        : ti.types.vector(3, ti.types.f32)
    new_orientation_2     : ti.types.vector(4, ti.types.f32)


@ti.func
def limit_angle(n : ti.types.vector(3, ti.types.f32),
                n1 : ti.types.vector(3, ti.types.f32),
                n2 : ti.types.vector(3, ti.types.f32),
                alpha : ti.types.f32,
                beta : ti.types.f32,
                delta_q : ti.types.vector(3, ti.types.f32)
                ) :
    """
        This function calculates whether the angle between two vectors is within the limits or not
        and if not, it calculates the delta rotation that needs to be applied to satisfy the limits.

        Source : Algorithm 3 https://matthias-research.github.io/pages/publications/PBDBodies.pdf

        Note: Instead of calculating phi from the arcsin, we use the atan2 function.

        Arguments:
        ----------
        n : ti.types.vector(3, ti.types.f32)
            -> Common rotation axis
        n1 : ti.types.vector(3, ti.types.f32)
            -> Rotation axis of object 1
        n2 : ti.types.vector(3, ti.types.f32)
            -> Rotation axis of object 2
        alpha : ti.types.f32
            -> Lower limit
        beta : ti.types.f32
            -> Upper limit
        delta_q : ti.types.vector(3, ti.types.f32)
            -> Delta rotation
        
    """

    phi = tm.atan2(tm.dot(tm.cross(n1, n2), n), tm.dot(n1, n2))

    if phi < alpha or phi > beta:
            
        phi = tm.clamp(phi, alpha, beta)

        rot = quaternion.from_axis_angle(n, phi)

        n1 = quaternion.rotate_vector(rot, n1)

        delta_q = tm.cross(n1, n2)

        return True, delta_q

    else:

        return False, delta_q

@ti.func
def compute_hinge_joint_constraint(
    obj_1               : ti.template(),
    obj_2               : ti.template(),
    h                   : ti.types.f32,
    constraint          : HingeJointConstraint,
):

    """
        The hinge joint constraint is a combination of three (or two) base constraints.

        The first constraint is an angular constraint to make sure the aligned axis of the two 
        objects is aligned.

        The second constraint is a position constraint to ensure the distance between the two bodies 
        is correct.

        The third constraint is an angular constraint to make sure that the angle limits are not 
        exceeded.

        Arguments:
        ----------
        obj_1 : ti.template()
            -> First object
        obj_2 : ti.template()
            -> Second object
        constraint : HingeJointConstraint
            -> Hinge joint constraint
        h : ti.types.f32
            -> Time step (substep)
    """

    force = ti.Vector([0.0, 0.0, 0.0])
    torque = ti.Vector([0.0, 0.0, 0.0])
    # Compute the aligned axis in the world coordinates

    e1_aligned_axis_wc = quaternion.rotate_vector(obj_1.orientation, constraint.e1_aligned_axis)
    e2_aligned_axis_wc = quaternion.rotate_vector(obj_2.orientation, constraint.e2_aligned_axis)

    # Compute the delta rotation between the two objects

    delta_q = tm.cross(e1_aligned_axis_wc, e2_aligned_axis_wc)

    # Apply the angular constraint

    aligned_constraint_response = compute_angular_constraint(obj_1, 
                                                              obj_2, 
                                                              h, 
                                                              tm.normalize(delta_q), 
                                                              tm.length(delta_q), 
                                                              constraint.lambda_aligned)

    obj_1.orientation = aligned_constraint_response.new_orientation_1
    obj_2.orientation = aligned_constraint_response.new_orientation_2
    torque += aligned_constraint_response.torque

    # Compute the position constraint
    r1_wc = quaternion.rotate_vector(obj_1.orientation, constraint.r1_lc)
    r2_wc = quaternion.rotate_vector(obj_2.orientation, constraint.r2_lc)

    p1 = obj_1.position + r1_wc
    p2 = obj_2.position + r2_wc

    delta_x = p2 - p1


    pos_constraint_response = compute_positional_constraint(obj_1, 
                                            constraint.r1_lc, 
                                            obj_2, constraint.r2_lc, 
                                            h, tm.normalize(delta_x), 
                                            tm.length(delta_x),
                                            constraint.lambda_positional)

    # Apply the position constraint
    obj_1.position    = pos_constraint_response.new_position_1
    obj_2.position    = pos_constraint_response.new_position_2
    obj_1.orientation = pos_constraint_response.new_orientation_1
    obj_2.orientation = pos_constraint_response.new_orientation_2
    force += pos_constraint_response.force

    
    if constraint.limited:
        # Compute the limit axis in the world coordinates

        e1_limit_axis_wc = quaternion.rotate_vector(obj_1.orientation, constraint.e1_limit_axis)
        e2_limit_axis_wc = quaternion.rotate_vector(obj_2.orientation, constraint.e2_limit_axis)

        n = e1_aligned_axis_wc

        # Compute the delta rotation between the two objects

        delta_q = tm.cross(e1_limit_axis_wc, e2_limit_axis_wc)

        over_limit, delta_q = limit_angle(n, e1_limit_axis_wc, 
                                          e2_limit_axis_wc, 
                                          constraint.lower_limit, constraint.upper_limit, delta_q)

        if over_limit:
            limit_constraint_response = compute_angular_constraint(obj_1, 
                                                                   obj_2, 
                                                                   h, 
                                                                   tm.normalize(delta_q), 
                                                                   tm.length(delta_q), 
                                                                   constraint.lambda_limit)
        
            obj_1.orientation = limit_constraint_response.new_orientation_1
            obj_2.orientation = limit_constraint_response.new_orientation_2
            torque += limit_constraint_response.torque

            

    return HingeJointConstraintResponse(
        force              =  force,
        torque             =  torque, 
        new_position_1     = obj_1.position,
        new_position_2     = obj_2.position,
        new_orientation_1  = obj_1.orientation,
        new_orientation_2  = obj_2.orientation,
        lambda_aligned     = aligned_constraint_response.lambda_aligned,
        lambda_positional  = pos_constraint_response.lambda_positional,
        lambda_limit       = limit_constraint_response.lambda_limit if constraint.limited else 0.0  
    )



