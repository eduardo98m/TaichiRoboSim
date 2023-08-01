
import taichi as ti
import taichi.math as tm

from bodies import RigidBody
from quaternion import quaternion

from .base_constraints import compute_positional_constraint, ConstraintResponse

@ti.dataclass
class CollisionPairConstraint:
    """
        Structure to store the collision constraints between two bodies.
    """
    body_1_idx     : ti.types.i32
    body_2_idx     : ti.types.i32
    
    collision      : bool
    
    lambda_normal  : ti.types.f32
    lambda_tangent : ti.types.f32
    
    static_friction_coeff  : ti.types.f32 = 0.5
    dynamic_friction_coeff : ti.types.f32 = 0.3

    contact_normal         : ti.types.vector(3, float)
    penetration_depth      : ti.types.f32

    r_1 : ti.types.vector(3, float)
    r_2 : ti.types.vector(3, float)

    fricction_force : ti.types.vector(3, float)
    normal_force    : ti.types.vector(3, float)


@ti.func
def compute_contact_constraint(
    body_1: RigidBody,
    body_2: RigidBody,
    constraint: CollisionPairConstraint,
    h: ti.types.f32,
):
    if constraint.collision == False:
        return body_1, body_2, constraint
    

    # To handle a contact during the position solve we compute the contact 
    # positions on the two bodies at the current state and before the substep 
    # integration as :
    p_1 = body_1.position + quaternion.rotate_vector(body_1.orientation, constraint.r_1)
    p_2 = body_2.position + quaternion.rotate_vector(body_2.orientation, constraint.r_2)

    p_1_prev = body_1.prev_position + quaternion.rotate_vector(body_1.prev_orientation, constraint.r_1)
    p_2_prev = body_2.prev_position + quaternion.rotate_vector(body_2.prev_orientation, constraint.r_2)

    #" If the bodies are penetrating we apply Δx = dn using alpha = 0 and lambda_normal"
    contact_normal_response : ConstraintResponse = compute_positional_constraint(
        body_1, 
        constraint.r_1, 
        body_2, 
        constraint.r_2,  
        h, 
        constraint.contact_normal, 
        constraint.penetration_depth, 
        constraint.lambda_normal,
        0
    )

    # Update the lambda_normal
    constraint.lambda_normal = contact_normal_response.new_lagrange_mult
    constraint.normal_force  = contact_normal_response.force

    # Update the position of the bodies
    body_1.position  = contact_normal_response.new_position_1
    body_2.position  = contact_normal_response.new_position_2

    body_1.orientation = contact_normal_response.new_orientation_1
    body_2.orientation = contact_normal_response.new_orientation_2

    # "To handle static friction we compute the relative motion of the contact points and its tangential component"
    n = constraint.contact_normal

    delta_p = (p_1 - p_1_prev) - (p_2 - p_2_prev)

    delta_p_t = delta_p - tm.dot(delta_p, n) * n 

    friction_direction = delta_p_t.normalize()
    friction_magnitude = tm.length(delta_p_t)

    if constraint.lambda_normal * constraint.static_friction_coeff > constraint.lambda_tangent:
    
        contact_friction_response : ConstraintResponse = compute_positional_constraint(
            body_1,
            constraint.r_1,
            body_2,
            constraint.r_2,
            h,
            friction_direction,
            friction_magnitude,
            constraint.lambda_tangent,
            0
        )

        # Update the lambda_tangent
        constraint.lambda_tangent = contact_friction_response.new_lagrange_mult
        constraint.fricction_force = contact_friction_response.force
    

    return body_1, body_2, constraint


@ti.func
def compute_contact_constraint_velocities(
    body_1: RigidBody,
    body_2: RigidBody,
    constraint: CollisionPairConstraint,
    h: ti.types.f32,
):  
    if constraint.collision == False:
        return 0
    
    # For each contact pair we compute the relative normal and tangential 
    # velocities at the contact point as : 
    v_1 = body_1.velocity
    w_1 = body_1.angular_velocity

    v_2 = body_2.velocity
    w_2 = body_2.angular_velocity

    # Rotate the relative position vectors to the world frame
    r_1_wc = body_1.position + quaternion.rotate_vector(body_1.orientation, constraint.r_1)
    r_2_wc = body_2.position + quaternion.rotate_vector(body_2.orientation, constraint.r_2)

    v_rel = (v_1 + tm.cross(w_1, r_1_wc)) - (v_2 + tm.cross(w_2, r_2_wc))

    v_rel_n = tm.dot(v_rel, constraint.contact_normal)

    v_rel_t = v_rel - v_rel_n * constraint.contact_normal

    # The friction force is integrated explicitly by computing the velocity update

    f_n = constraint.lambda_normal / h**2

    delta_v = - v_rel_t/abs(v_rel_t) * tm.min(h * constraint.dynamic_friction_coeff * constraint.lambda_normal, tm.length(v_rel_t))

    
    
        



    


