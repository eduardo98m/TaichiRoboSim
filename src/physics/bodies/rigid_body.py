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
    mass                : ti.types.f32
    inertia             : ti.types.matrix(3,3, float) 
    position            : ti.types.vector(3,   float)
    prev_position       : ti.types.vector(3,   float)
    velocity            : ti.types.vector(3,   float)
    external_force      : ti.types.vector(3,   float) 
    orientation         : ti.types.vector(4,   float)
    prev_orientation    : ti.types.vector(4,   float) 
    angular_velocity    : ti.types.vector(3,   float)
    external_torque     : ti.types.vector(3,   float)
    inv_inertia         : ti.types.matrix(3,3, float)
    dynamic_inv_interia : ti.types.matrix(3,3, float)
    fixed               : bool
    collider            : Collider
    collision_group     : ti.types.i32
    material            : Material

    def init(self):
        self.prev_position = self.position
        self.prev_orientation = self.orientation

    
    @ti.func
    def compute_inv_inertia(self):
        self.inv_inertia = tm.inverse(self.inertia)

    @ti.func
    def compute_dynamic_inv_inertia(self):
       rotation_matrix          = quaternion.to_rotation_matrix(self.orientation)       
       self.dynamic_inv_interia = rotation_matrix @ self.inv_inertia @ rotation_matrix.transpose()
    
    @ti.func
    def compute_transformation_matrix(self) -> ti.types.matrix(4,4, float):

        # Create the rotation matrix
        rotation_matrix = quaternion.to_rotation_matrix(self.orientation)

        # Create the transformation matrix
        transformation_matrix = ti.Matrix.identity(ti.f32, 4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3]   = self.position

        return transformation_matrix
        

    
    