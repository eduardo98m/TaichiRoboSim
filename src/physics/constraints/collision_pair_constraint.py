
import taichi as ti
import taichi.math as tm

from bodies import RigidBody
from quaternion import quaternion

from .constraint import Constraint, ConstraintResponse
from .positional_constraint import compute_positional_constraint, positional_contraint_generalized_inverse_mass


EPSILON = 1e-5
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
    restitution_coeff : ti.types.f32 = 0.5

    contact_normal         : ti.types.vector(3, float)
    penetration_depth      : ti.types.f32

    r_1 : ti.types.vector(3, float)
    r_2 : ti.types.vector(3, float)

    fricction_force : ti.types.vector(3, float)
    normal_force    : ti.types.vector(3, float)

    def init(self, body_1: RigidBody, body_2: RigidBody):
        self.static_friction_coeff = (body_1.material.static_friction_coeff + body_2.material.static_friction_coeff)/2
        self.dynamic_friction_coeff = (body_1.material.dynamic_friction_coeff + body_2.material.dynamic_friction_coeff)/2
        self.restitution_coeff = (body_1.material.restitution_coeff + body_2.material.restitution_coeff)/2
        self.collision  = False
    
    @ti.func
    def reset_lambda(self):
        self.lambda_normal = 0.0
        self.lambda_tangent = 0.0


@ti.dataclass
class ContactConstraintResponse:
    """
        Structure to store the response of a contact constraint.
    """
    body_1 : RigidBody
    body_2 : RigidBody
    contact_constraint : CollisionPairConstraint


@ti.dataclass
class ContactVelocityConstraintResponse:
    """
        Structure to store the response of a contact constraint.
    """
    body_1 : RigidBody
    body_2 : RigidBody

@ti.func
def compute_contact_constraint(
    body_1: RigidBody,
    body_2: RigidBody,
    constraint: CollisionPairConstraint,
    h: ti.types.f32
):  
    
        
    # To handle a contact during the position solve we compute the contact 
    # positions on the two bodies at the current state and before the substep 
    # integration as :
    p_1 = body_1.position + quaternion.rotate_vector(body_1.orientation, constraint.r_1)
    p_2 = body_2.position + quaternion.rotate_vector(body_2.orientation, constraint.r_2)

    p_1_prev = body_1.prev_position + quaternion.rotate_vector(body_1.prev_orientation, constraint.r_1)
    p_2_prev = body_2.prev_position + quaternion.rotate_vector(body_2.prev_orientation, constraint.r_2)

    complementary_constraint_data = Constraint(
        direction     = constraint.contact_normal,
        magnitude     = constraint.penetration_depth,
        lagrange_mult = constraint.lambda_normal,
        compliance    = 0
    ) 
    

    #" If the bodies are penetrating we apply Δx = dn using alpha = 0 and lambda_normal"
    complementary_constraint_response = compute_positional_constraint(
        body_1, 
        body_2, 
        complementary_constraint_data,
        h
    )

    # Update the lambda_normal
    constraint.lambda_normal = complementary_constraint_response.new_lagrange_mult
    constraint.normal_force  = complementary_constraint_response.force
    # Update the position of the bodies
    body_1.position  = complementary_constraint_response.new_position_1
    body_2.position  = complementary_constraint_response.new_position_2

    body_1.orientation = complementary_constraint_response.new_orientation_1
    body_2.orientation = complementary_constraint_response.new_orientation_2


    # "To handle static friction we compute the relative motion of the contact points and its tangential component"
    if constraint.lambda_normal * constraint.static_friction_coeff > constraint.lambda_tangent:

        n = constraint.contact_normal

        delta_p = (p_1 - p_1_prev) - (p_2 - p_2_prev)

        delta_p_t = delta_p - tm.dot(delta_p, n) * n 

        friction_direction = tm.normalize(delta_p_t)
        friction_magnitude = tm.length(delta_p_t)

        friction_constraint_data = Constraint(
            direction     = friction_direction,
            magnitude     = friction_magnitude,
            lagrange_mult = constraint.lambda_tangent,
            compliance    = 0.0
        ) 
    
        friction_constraint_response = compute_positional_constraint(
            body_1,
            body_2,
            friction_constraint_data,
            h
        )

        # Update the lambda_tangent
        constraint.lambda_tangent  = friction_constraint_response.new_lagrange_mult
        constraint.fricction_force = friction_constraint_response.force

        body_1.position  = friction_constraint_response.new_position_1
        body_2.position  = friction_constraint_response.new_position_2

        body_1.orientation = friction_constraint_response.new_orientation_1
        body_2.orientation = friction_constraint_response.new_orientation_2
        

    return ContactConstraintResponse(body_1, body_2, constraint)


@ti.func
def compute_contact_velocity_constraint(
    body_1: RigidBody,
    body_2: RigidBody,
    constraint: CollisionPairConstraint,
    h: ti.types.f32,
    gravity_acceleration: ti.types.f32
):  
    
    response = ContactVelocityConstraintResponse(body_1, body_2)

    if constraint.collision:
    
        # For each contact pair we compute the relative normal and tangential 
        # velocities at the contact point as : 
        v_1 = body_1.velocity
        w_1 = body_1.angular_velocity

        v_2 = body_2.velocity
        w_2 = body_2.angular_velocity

        n = constraint.contact_normal

        # Rotate the relative position vectors to the world frame
        r_1_wc = body_1.position + quaternion.rotate_vector(body_1.orientation, constraint.r_1)
        r_2_wc = body_2.position + quaternion.rotate_vector(body_2.orientation, constraint.r_2)

        v_rel = (v_1 + tm.cross(w_1, r_1_wc)) - (v_2 + tm.cross(w_2, r_2_wc))

        v_rel_n = tm.dot(v_rel, n)

        v_rel_t = v_rel - v_rel_n * n


        # The friction force is integrated explicitly by computing the velocity update
        f_n = constraint.lambda_normal / h**2
        v_rel_t_norm = ti.Vector.zero(ti.f32, 3)
        if tm.length(v_rel_t) > 0.0:
            v_rel_t_norm = tm.normalize(v_rel_t)
        delta_v = - v_rel_t_norm * \
                    tm.min(h * constraint.dynamic_friction_coeff * abs(f_n), 
                        tm.length(v_rel_t))
        
        # Now we handle restitution
        v_1_prev  = body_1.prev_velocity
        v_2_prev  = body_2.prev_velocity
        w_1_prev  = body_1.prev_angular_velocity
        w_2_prev  = body_2.prev_angular_velocity
        
        v_rel_til = (v_1_prev + tm.cross(w_1_prev, r_1_wc)) - (v_2_prev + tm.cross(w_2_prev, r_2_wc))

        v_rel_n_til = tm.dot(v_rel_til, n)

        e = 0.0
        if abs(v_rel_n) >= 2 * h * gravity_acceleration:   
            e = constraint.restitution_coeff

        # We update  the delta_v with the restitution term
        delta_v = delta_v - n * (- v_rel_n +  tm.min(-e*v_rel_n_til, 0 ) ) 
        # We apply the delta_v to the bodies
        body_1.compute_dynamic_inv_inertia()
        body_2.compute_dynamic_inv_inertia()

        inv_inertia_1 = body_1.dynamic_inv_interia
        inv_inertia_2 = body_2.dynamic_inv_interia

        gen_inv_mass_1 = 0.0
        if not body_1.fixed:
            gen_inv_mass_1 = positional_contraint_generalized_inverse_mass(
                body_1.mass,
                inv_inertia_1,
                r_1_wc,
                n
            )
        gen_inv_mass_2 = 0.0
        if not body_2.fixed:
            gen_inv_mass_2 = positional_contraint_generalized_inverse_mass(
                body_2.mass,
                inv_inertia_2,
                r_1_wc,
                n
            )
        p = -delta_v / (gen_inv_mass_1 + gen_inv_mass_2) 
        
        if not body_1.fixed:
            body_1.velocity = body_1.velocity +  p / body_1.mass
            body_1.angular_velocity = body_1.angular_velocity + inv_inertia_1 @ tm.cross(r_1_wc, p)
        
        if not body_2.fixed:
            body_2.velocity = body_2.velocity -  p / body_2.mass  
            body_2.angular_velocity = body_2.angular_velocity - inv_inertia_2 @ tm.cross(r_2_wc, p)
        
        response = ContactVelocityConstraintResponse(body_1, body_2)
        

    return response





    
    
        



    


