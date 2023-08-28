import taichi as ti
import taichi.math as tm

from quaternion import quaternion
from collision import Collider

from .material import Material


@ti.dataclass
class RigidBody:
    """
    Class for all objects that can move and collide with each other
    
    Arguments:
    ----------
    * mass : ti.types.f32
        Mass of the object
    * inertia : ti.types.matrix(3,3, float)
        Inertia tensor of the object
    * position : ti.types.vector(3, float)
        Position of the object in the world
    * velocity : ti.types.vector(3, float)
        Velocity of the object
    * orientation : ti.types.vector(4, float)
        Orientation of the object in the world as a quaternion
    * angular_velocity : ti.types.vector(3, float)
        Angular velocity of the object
    * collider_idx: Collider
        Index of the collider that is attached to the object
    * `material` : Material
        -> Struc that stores the material properties of the object
    """
    mass                  : ti.float32
    inertia               : ti.types.matrix(3,3, ti.float32) 
    position              : ti.types.vector(3,   ti.float32)
    prev_position         : ti.types.vector(3,   ti.float32)
    velocity              : ti.types.vector(3,   ti.float32)
    prev_velocity         : ti.types.vector(3,   ti.float32)
    external_force        : ti.types.vector(3,   ti.float32) 
    orientation           : ti.types.vector(4,   ti.float32)
    prev_orientation      : ti.types.vector(4,   ti.float32) 
    angular_velocity      : ti.types.vector(3,   ti.float32)
    prev_angular_velocity : ti.types.vector(3,   ti.float32)
    external_torque       : ti.types.vector(3,   ti.float32)
    inv_inertia           : ti.types.matrix(3,3, ti.float32)
    dynamic_inertia       : ti.types.matrix(3,3, ti.float32)
    dynamic_inv_interia   : ti.types.matrix(3,3, ti.float32)
    fixed                 : bool
    collider              : Collider
    collision_group       : ti.int32
    material              : Material
    f_ext                 : ti.types.vector(3, ti.float32)
    t_ext                 : ti.types.vector(3, ti.float32)

    def init(self):
        self.prev_position = self.position
        self.prev_orientation = self.orientation

    
    @ti.func
    def compute_inv_inertia(self):
        self.inv_inertia = tm.inverse(self.inertia)

    @ti.func
    def update_inertia(self):
        """
            Updates the dynamic inertia tensor of the body
        """
        if not self.fixed:
            rotation_matrix          = quaternion.to_rotation_matrix(self.orientation)       
            self.dynamic_inv_interia = rotation_matrix @ self.inv_inertia @ rotation_matrix.transpose()
            self.dynamic_inertia     = rotation_matrix @ self.inertia @ rotation_matrix.transpose()
        else:
            self.dynamic_inv_interia = ti.Matrix.zero(ti.float32, 3, 3)
            self.dynamic_inertia     = ti.Matrix.zero(ti.float32, 3, 3)
    
    @ti.func
    def position_update(self, h: ti.float32):
        """
        Update the position of the object, a
        
        """
        if not self.fixed:
            self.prev_position         = self.position
            self.prev_orientation      = self.orientation
            self.prev_velocity         = self.velocity
            self.prev_angular_velocity = self.angular_velocity
            self.update_inertia()

            # Update velocity
            self.velocity    = self.velocity + (self.f_ext/ self.mass) * h 
            
            # Update position
            self.position    = self.position + self.velocity * h 
            
            # Update angular velocity
            self.angular_velocity = self.angular_velocity +  h * self.dynamic_inv_interia @ (
                self.t_ext - tm.cross(self.angular_velocity, self.dynamic_inertia @ self.angular_velocity)
            )
            # Update orientation
            self.orientation = quaternion.rotate_by_axis(self.orientation, self.angular_velocity * h)
    
    
    @ti.func
    def velocity_update(self, h: ti.float32):
        """
            Updates the velocity of the body.
        """

        if not self.fixed:
            self.velocity = (self.position - self.prev_position) / h

            delta_orientation = quaternion.hamilton_product(self.orientation, 
                                            quaternion.inverse(self.prev_orientation))
            

            angular_velocity  = 2.0 * ti.Vector([   delta_orientation[1],
                                                    delta_orientation[2],
                                                    delta_orientation[3]
                                                ])/  h

            if delta_orientation[0] >=  0.0: 
                self.angular_velocity = - angular_velocity
            else:
                self.angular_velocity = angular_velocity

    
    @ti.func
    def compute_positional_generalized_inverse_mass(self, 
                                                    r:ti.types.vector(3, ti.float32), 
                                                    n:ti.types.vector(3, ti.float32)) -> ti.float32:
        """
            Arguments:
            ----------
            * `r` : ti.types.vector(3, ti.float32)
                -> Vector relative to the center of mass of the object, where the constraint is applied.
            * `n` : ti.types.vector(3, ti.float32)
                -> Normal vector of the constraint
        """
        w = 0.0
        if not self.fixed:
            self.update_inertia()
            w =  1 / self.mass + tm.dot(tm.cross(r, n), self.dynamic_inv_interia @ tm.cross(r, n))
        return w
    
    @ti.func
    def apply_position_correction(self, p : ti.types.vector(3, ti.float32), r: ti.types.vector(3, ti.float32)):
        """
            Arguments:
            ---------
            * `p`: ti.types.vector(3, ti.float32)
                -> Positional correction // Impulse
            * `r` : ti.types.vector(3, ti.float32)
                -> Position of the constraint relative to the object mass center
        """
        if not self.fixed:
            # NOTE: We do not update the inertia as it is already updated when the generalized inverse mass is calculated
            self.position = self.position + p/self.mass
            self.orientation = quaternion.rotate_by_axis(self.orientation, self.dynamic_inv_interia @ tm.cross(r, p))
    
    @ti.func
    def apply_angular_correction(self, p : ti.types.vector(3, ti.float32)):
        """
            Arguments:
            ---------
            * `p`: ti.types.vector(3, ti.float32)
                -> Angular correction // Impulse
        """
        if not self.fixed:
            self.orientation = quaternion.rotate_by_axis(self.orientation, self.dynamic_inv_interia @ p)

    
    @ti.func
    def compute_angular_generalize_inverse_mass(self, n: ti.types.vector(3, ti.float32)) -> ti.float32:
        """
            Arguments:
            ----------
            n : ti.types.vector(3, ti.types.f32)
                -> Direction of the constraint (rotation axis)
        """
        w = 0.0
        if not self.fixed:
            self.update_inertia()
            w = tm.dot(n, self.dynamic_inv_interia @ n)
        return w
    
    def apply_gravity(self, gravity: ti.types.vector(3, ti.float32)):
        """
            Computes the gravitational force applied to the object

            Note: This should only be called once in the simulation as the gravity is constant

            Arguments:
            ----------

            gravity : ti.types.vector(3, ti.float32)
                -> Gravity Vector (In world coordiantes)
        """
        self.f_ext += self.mass * gravity
        #self.t_ext += tm.cross(self.position, self.mass * gravity)

    @ti.func
    def clear_forces_and_torques(self):
        """
            Clears the forces and torques applied to the object

            By applying this function, the gravity will not be applied to the object, so you should 
            call apply_gravity() after calling this function.
        """
        self.f_ext = ti.Vector.zero(ti.float32, 3)
        self.t_ext = ti.Vector.zero(ti.float32, 3)


    
    @ti.func
    def compute_transformation_matrix(self) -> ti.types.matrix(4,4, float):

        # Create the rotation matrix
        rotation_matrix = quaternion.to_rotation_matrix(self.orientation)

        # Create the transformation matrix
        transformation_matrix = tm.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3]   = self.position

        return transformation_matrix
        

    
    