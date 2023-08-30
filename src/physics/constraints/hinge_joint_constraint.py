import taichi as ti
import taichi.math as tm

from quaternion import quaternion

from .constraint import Constraint, PositionCorrection
from .angular_constraint    import compute_angular_constraint, angular_constraint_velocity_update
from .positional_constraint import compute_positional_constraint
from .angle_limit import angle_limit

from bodies import RigidBody

@ti.dataclass
class HingeJointConstraint:

    body_1_idx : ti.types.i32
    body_2_idx : ti.types.i32

    axes_1 : ti.types.matrix(3, 3, ti.f32)
    axes_2 : ti.types.matrix(3, 3, ti.f32)

    r_1 : ti.types.vector(3, ti.float32)
    r_2 : ti.types.vector(3, ti.float32)

    compliance          : ti.f32
    damping             : ti.f32
	
    aligned_constraint     : Constraint
    angle_limit_constraint : Constraint
    drive_joint_constraint : Constraint
    attachment_point_constraint : Constraint
    
    driven : bool
    drive_by_speed : bool
    target_speed   : ti.types.f32
    target_angle   : ti.types.f32

    # Used for limiting the angle
    limited : bool
    lower_limit  : ti.types.f32
    upper_limit  : ti.types.f32

    force  : ti.types.vector(3, ti.f32,)
    torque : ti.types.vector(3, ti.f32,)

    current_angle : ti.types.f32

    def initialize(self):

        self.aligned_constraint = Constraint(
            lagrange_mult   = 0.0,
            compliance      = 0.0,
            r_1             = self.r_1,
            r_2             = self.r_2,
            direction       = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
            magnitude       = 0.0
        )

        self.angle_limit_constraint = Constraint(
            lagrange_mult   = 0.0,
            compliance      = 0.0,
            r_1             = self.r_1,
            r_2             = self.r_2,
            direction       = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
            magnitude       = 0.0
        )

        self.drive_joint_constraint = Constraint(
            lagrange_mult   = 0.0,
            compliance      = self.compliance,
            r_1             = self.r_1,
            r_2             = self.r_2,
            direction       = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
            magnitude       = 0.0
        )

        self.attachment_point_constraint = Constraint(
            lagrange_mult   = 0.0,
            compliance      = 0.0,
            r_1             = self.r_1,
            r_2             = self.r_2,
            direction       = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32),
            magnitude       = 0.0
        )
    
    def drive_joint_pos(self, target_angle):
        self.driven = True
        self.target_angle = target_angle
    
    def drive_joint_vel(self, target_vel):
        self.driven = True
        self.target_vel = target_vel

    def limit_joint(self, lower_limit, upper_limit):
        self.limited = True
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit


    @ti.func
    def solve_position(
        self,
        body_1              : RigidBody,
        body_2              : RigidBody,
        h                   : ti.types.f32,
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
            `body_1` : RigidBody
                -> First object
            `body_2` : RigidBody
                -> Second object
            `constraint` : HingeJointConstraint
                -> Hinge joint constraint
            `h` : ti.types.f32
                -> Time step (substep)
        """

        force  = ti.Vector.zero(ti.f32, 3)
        torque = ti.Vector.zero(ti.f32, 3)

        axes_1 = quaternion.to_rotation_matrix(body_1.orientation)
        axes_2 = quaternion.to_rotation_matrix(body_2.orientation)

        a_1 = axes_1[0, :]
        a_2 = axes_2[0, :]
        b_1 = axes_1[1, :]
        b_2 = axes_2[1, :]

        # Compute the delta rotation between the two objects

        delta_q = tm.cross(a_1, a_2)

        # self.aligned_constraint.magnitude = tm.length(delta_q)

        # if self.aligned_constraint.magnitude == 0.0:
        #     self.aligned_constraint.direction = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        # else:
        #     self.aligned_constraint.direction = tm.normalize(delta_q)
        

        # # Apply the angular constraint

        # aligned_constraint_response = compute_angular_constraint(body_1, 
        #                                                         body_2,  
        #                                                         self.aligned_constraint,
        #                                                         h)

        # body_1.orientation = aligned_constraint_response.new_orientation_1
        # body_2.orientation = aligned_constraint_response.new_orientation_2
        # self.aligned_constraint.lagrange_mult = aligned_constraint_response.new_lagrange_mult
        # torque += aligned_constraint_response.torque


        # Compute the attachment constraint
        r_1_wc = quaternion.rotate_vector(body_1.orientation, self.r_1)
        r_2_wc = quaternion.rotate_vector(body_2.orientation, self.r_2)

        p_1 = body_1.position + r_1_wc
        p_2 = body_2.position + r_2_wc

        delta_x = p_1 - p_2

        self.attachment_point_constraint.magnitude = tm.length(delta_x)

        if self.attachment_point_constraint.magnitude == 0.0:
            self.attachment_point_constraint.direction = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        else:
            self.attachment_point_constraint.direction = tm.normalize(delta_x)
        
        attachment_point_constraint_response = compute_positional_constraint(body_1, 
                                                                body_2, 
                                                                self.attachment_point_constraint, 
                                                                h)

        # Apply the position constraint
        body_1.position    = attachment_point_constraint_response.new_position_1
        body_2.position    = attachment_point_constraint_response.new_position_2
        body_1.orientation = attachment_point_constraint_response.new_orientation_1
        body_2.orientation = attachment_point_constraint_response.new_orientation_2
        self.attachment_point_constraint.lagrange_mult = attachment_point_constraint_response.new_lagrange_mult


        force += attachment_point_constraint_response.force

        
        # TODO : This is also calculated in the angle_limit.py file and can be reused
        self.current_angle = tm.atan2(tm.dot(tm.cross(b_1, b_2), a_1), tm.dot(b_1, b_2))
        
        
        if self.limited:
            over_limit, delta_q = angle_limit(a_1, b_1, b_2,  
                                            self.lower_limit, 
                                            self.upper_limit)
            
            self.angle_limit_constraint.direction = tm.normalize(delta_q)
            self.angle_limit_constraint.magnitude = tm.length(delta_q)

            if over_limit:
                limit_constraint_response = compute_angular_constraint(
                                                                    body_1, 
                                                                    body_2, 
                                                                    self.angle_limit_constraint,
                                                                    h)
            
                body_1.orientation = limit_constraint_response.new_orientation_1
                body_2.orientation = limit_constraint_response.new_orientation_2
                torque += limit_constraint_response.torque
        
        if self.driven:
            target_angle = self.target_angle

            if self.drive_by_speed:
                target_angle = self.current_angle + self.target_speed * h
            
            
            b_target = b_1 * tm.cos(target_angle) + \
                    tm.cross(a_1, b_1) * tm.sin(target_angle) + \
                    a_1 * tm.dot(a_1, b_1) * (1 - tm.cos(target_angle))

            delta_q_target =  tm.cross(b_target, b_2)

            self.drive_joint_constraint.direction = tm.normalize(delta_q_target)
            self.drive_joint_constraint.magnitude = tm.length(delta_q_target)

            drive_joint_constraint_response = compute_angular_constraint(body_1,
                                                                            body_2,
                                                                            self.drive_joint_constraint,
                                                                            h)
            
            body_1.orientation = drive_joint_constraint_response.new_orientation_1
            body_2.orientation = drive_joint_constraint_response.new_orientation_2
            torque += drive_joint_constraint_response.torque

        self.torque = torque
        self.force  = force
        return PositionCorrection(
            new_position_1     = body_1.position,
            new_position_2     = body_2.position,
            new_orientation_1  = body_1.orientation,
            new_orientation_2  = body_2.orientation
        )
    
    @ti.func
    def solve_velocity(
        self, 
        body_1              : RigidBody,
        body_2              : RigidBody,
        h                   : ti.types.f32
    ):
        """
            Applies the velocity update for the hinge joint constraint. Applies the the joint damping.

            Arguments:
            ---------- 
            `body_1` : RigidBody
                -> First object
            `body_2` : RigidBody
                -> Second object
            `h` : ti.types.f32
                -> Time step (substep)
                
        """
        delta_omega = (body_2.angular_velocity - body_1.angular_velocity) * tm.min(self.damping * h, 1.0)
        
        response = angular_constraint_velocity_update(
            body_1,
            body_2,
            delta_omega,
            h
        )
        self.torque = self.torque + response.torque

        return PositionCorrection(
            new_orientation_1  = response.new_orientation_1,
            new_orientation_2  = response.new_orientation_2
        )
        
