import taichi as ti
import taichi.math as tm
#from colliders import Collider, SphereCollider
from time import time, sleep
from quaternion import quaternion

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from renderer import render_heightfield, render_plane

from collision import colliders
from collision import BoxCollider, SphereCollider, CylinderCollider, HeightFieldCollider, PlaneCollider

from constraints import hinge_joint_constraint, constraint_types

from rigid_body import RigidBody

@ti.dataclass
class Material():
    """
        Class that stores the material properties of a body
    """
    static_friction_coeff  : ti.types.f32
    dynamic_friction_coeff : ti.types.f32


@ti.dataclass
class CollisionPairConstraint():
    """
        Class that stores the indices of the two objects that are colliding
    """
    body_1_idx     : ti.types.u32
    body_2_idx     : ti.types.u32
    
    collision      : bool
    
    lambda_normal  : ti.types.f32
    lambda_tangent : ti.types.f32
    
    static_friction_coeff  : ti.types.f32
    dynamic_friction_coeff : ti.types.f32

    contact_normal         : ti.types.vector(3, float)
    penetration_depth      : ti.types.f32

    r_1 : ti.types.vector(3, float)
    r_2 : ti.types.vector(3, float)

    fricction_force : ti.types.vector(3, float)
    normal_force    : ti.types.vector(3, float)



@ti.data_oriented
class PhysicsWorld():
    
    def __init__(self, 
                 dt : ti.types.f32,
                 n_substeps: ti.types.u32,
                 use_visualizer = True,
                 visualizer_port = "4343") -> None:
        
        self.gravity_vector   = ti.Vector([0.0, -9.8, 0.0])
        
        self.bodies_list                     = []
        
        self.colliders_list                  = []
        self.potential_collision_pairs_list  = []
        self.collision_groups_list           = []
        self.materials_list                  = []

        self.hinge_constraints_list          = []

        self.box_colliders_list              = []
        self.sphere_colliders_list           = []
        self.cyllinder_colliders_list        = []
        self.plane_colliders_list            = []

        self.heightfield_colliders_list      = []
        self.heightfield_x_coord_list        = []
        self.heightfield_y_coord_list        = []
        self.heightfield_data_list           = []

        self.n_bodies         = 0
        self.dt               = dt
        self.n_substeps       = n_substeps

        self.visualizer_active = use_visualizer
        if use_visualizer:
            zmq_url = "tcp://127.0.0.1:" + visualizer_port
            print("Visualizer running on URL: ", zmq_url)
            self.visualizer = meshcat.Visualizer(zmq_url=zmq_url)
            self.visualizer.open()
    

    def set_up_simulation(self):
        self.rigid_bodies               = RigidBody.field(shape=(self.n_bodies,))
        
        self.rigid_bodies_transformations  = tm.mat4.field(shape=(self.n_bodies,))
        # Add the rigid bodies in the rigid bodies field
        for i in range(self.n_bodies):
            self.rigid_bodies[i] = self.bodies_list[i] 
        
        n_cyllinder_colliders = len(self.cyllinder_colliders_list)
        self.cyllinder_colliders        = CylinderCollider.field(shape=(n_cyllinder_colliders,))
        for i in range(n_cyllinder_colliders):
            self.cyllinder_colliders[i] = self.cyllinder_colliders_list[i]
        
        n_sphere_colliders = len(self.sphere_colliders_list)
        self.sphere_colliders           = SphereCollider.field(shape=(n_sphere_colliders,))
        for i in range(n_sphere_colliders):
            self.sphere_colliders[i] = self.sphere_colliders_list[i]
        
        n_box_colliders = len(self.box_colliders_list)
        self.box_colliders              = BoxCollider.field(shape=(n_box_colliders,))
        for i in range(n_box_colliders):
            self.box_colliders[i] = self.box_colliders_list[i]

        n_plane_colliders = len(self.plane_colliders_list)
        self.plane_colliders            = PlaneCollider.field(shape=(n_plane_colliders,))
        for i in range(n_plane_colliders):
            self.plane_colliders[i] = self.plane_colliders_list[i]

        n_heightfield_colliders = len(self.heightfield_colliders_list)
        self.heightfield_colliders      = HeightFieldCollider.field(shape=(n_heightfield_colliders,))
        for i in range(n_heightfield_colliders):
            self.heightfield_colliders[i] = self.heightfield_colliders_list[i]
 
        self.precompute_collision_checks()

    
    def add_height_field(self, 
                         height_field : HeightFieldCollider,
                         x_coordinates,
                         y_coordinates,
                         heightfield_data):
        
        height_field.x_coordinates_ptr    = len(x_coordinates)
        height_field.y_coordinates_ptr    = len(y_coordinates)
        height_field.heightfield_data_ptr = len(heightfield_data)
        
        self.heightfield_x_coord_list.append(x_coordinates)
        self.heightfield_y_coord_list.append(y_coordinates)
        self.heightfield_data_list.append(heightfield_data)
        self.heightfield_colliders_list.append(height_field)
    

    def create_constraint(
            self,
            constraint_type : ti.types.u32,
            constraint            
    ):
        
        if constraint_type == constraint_types.HINGE_JOINT:
            self.hinge_constraints_list.append(constraint)
            constraint_idx = len(self.hinge_constraints_list) - 1
            return constraint_idx
        else:
            print("Constraint type not supported")
            return -1

    def set_gravity_vector(self, 
                           gravity_vector : ti.types.vector(3, float)):
        self.gravity_vector = gravity_vector
    
    def add_rigid_body(self,
                       body : RigidBody,
                       collider) -> ti.types.u32:
        

        if   body.collider_type == colliders.BOX:
            self.box_colliders_list.append(collider)
            body.collider_idx = len(self.box_colliders_list) - 1
        elif body.collider_type == colliders.SPHERE:
            self.sphere_colliders_list.append(collider)
            body.collider_idx = len(self.sphere_colliders_list) - 1
        elif body.collider_type == colliders.CYLLINDER:
            self.cyllinder_colliders_list.append(collider)
            body.collider_idx = len(self.cyllinder_colliders_list) - 1
        elif body.collider_type == colliders.PLANE:
            self.plane_colliders_list.append(collider)
            body.collider_idx = len(self.plane_colliders_list) - 1
        elif body.collider_type == colliders.HEIGHTFIELD:
            self.heightfield_colliders_list.append(collider)
            body.collider_idx = len(self.heightfield_colliders_list) - 1

        self.bodies_list.append(body)
        self.n_bodies += 1

        return self.n_bodies - 1

    @ti.func
    def update_rigid_bodies_position_and_orientation(
            self,
            h: ti.types.f32 
    ):
        """
            Integrate the dynamic bodies position, velocity, orientation and angular velocity
            using the Euler method.

            Taken from : Algorithm 2, pag 5.
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf

            Arguments:
            ----------
                h : float 
                    The time step (substep)
        """
        for i in range(self.rigid_bodies):
            obj = self.dynamic_objects[i]
            # Check if the object is static
            if obj.fixed: continue

            # Save the previous position
            obj.previous_position = obj.position

            # Update velocity
            obj.velocity    = obj.velocity + (self.gravity_vector  + obj.external_force/ obj.mass) * ti.static(h) 
            
            # Update position
            obj.position    = obj.position + obj.velocity  * ti.static(h) 

            # Save the previous orientation
            obj.previous_orientation = obj.orientation

            # Update angular velocity
            obj.angular_velocity = obj.angular_velocity +  ti.static(h) * obj.inv_inertia * (
                obj.external_torque - tm.cross(obj.angular_velocity, obj.inertia @ obj.angular_velocity)
            )

            # Update orientation
            obj.orientation = tm.normalize(
                obj.orientation + 0.5 * ti.static(h) * quaternion.rotate_vector(obj.orientation, obj.angular_velocity)
            )

    @ti.func
    def update_rigid_bodies_velocities(
        self,
        h : ti.types.f32
    ):
        """
            Computes the velocities (linear and angular) of the dynamic bodies.
            Based on the previous and current position and orientation. 

            Taken from : Algorithm 2, pag 5.
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf

            Arguments:
            ----------
                h : float 
                    The time step (substep)     
        """

        for i in range(self.rigid_bodies):
            if obj.fixed: continue

            obj          = self.dynamic_objects[i]
            obj.velocity = (obj.position - obj.previous_position) / ti.static(h)

            delta_orientation = quaternion.hamilton_product(obj.orientation, 
                                            quaternion.inverse(obj.previous_orientation))
            

            angular_velocity  = 2.0 * ti.Vector([   delta_orientation[1],
                                                    delta_orientation[2],
                                                    delta_orientation[3]
                                                ])/  ti.static(h)

            if delta_orientation[0] >=  0.0: 
                obj.angular_velocity = - angular_velocity
            else:
                obj.angular_velocity = angular_velocity
    

    def precompute_potential_collison_pairs(self):
        """
            Precompute the collision checks.

            This function is called once at the beginning of the simulation.

            It simply looks at the rigid bodies and checks for possible collision pairs, so the 
            real collision detection can be done in a more efficient way.
        """
        
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                body_1 = self.bodies_list[i]
                body_2 = self.bodies_list[j]

                if body_1.fixed and body_2.fixed: continue

                if self.collision_groups_list[body_1.collision_group, body_2.collision_group] == 0: continue

                self.potential_collision_pairs_list.append((i, j))
        
        # Create a field to store the collision pairs
        self.collision_pairs_constraints = ti.field(dtype=CollisionPairConstraint, shape=(len(self.potential_collision_pairs_list)))


        # Add the collision pairs to the field
        for i in range(len(self.potential_collision_pairs_list)):
            body_1_idx = self.potential_collision_pairs_list[i][0]
            body_2_idx = self.potential_collision_pairs_list[i][1]
            body_1 = self.bodies_list[body_1_idx]
            body_2 = self.bodies_list[body_2_idx]
            self.collision_pairs_constraints[i].body_1_idx = body_1_idx
            self.collision_pairs_constraints[i].body_2_idx = body_2_idx
            self.collision_pairs_constraints[i].collision  = False
            self.collision_pairs_constraints[i].static_friction_coefficient  = \
                (body_1.static_friction_coefficient + body_2.static_friction_coefficient)/2
            self.collision_pairs_constraints[i].dynamic_friction_coefficient = \
                (body_1.dynamic_friction_coefficient + body_2.dynamic_friction_coefficient)/2

    @ti.func
    def collect_collision_pairs(self):
        """
            Collect the collision pairs.
        """
        
        for i in range(self.collision_pairs_constraints.shape[0]):

            constraint = self.collision_pairs_constraints[i]

            body_1 = self.bodies_list[constraint.body_1_idx]
            
            body_1_collider = self.get_collider_info(body_1.collider_type, body_1.collider_idx)

            body_2 = self.bodies_list[constraint.body_2_idx]

            body_2_collider = self.get_collider_info(body_2.collider_type, body_2.collider_idx)

            aabb_safety_expansion_1 = ti.max(tm.length(body_1.velocity) * self.dt * 2.0, 1.0)
            aabb_safety_expansion_2 = ti.max(tm.length(body_2.velocity) * self.dt * 2.0, 1.0)

            broad_phase_check = self.broad_phase_collision_detection(
                body_1.position, body_1.orientation, body_1_collider.aabb * aabb_safety_expansion_1,
                body_2.position, body_2.orientation, body_2_collider.aabb * aabb_safety_expansion_2
            )

            if broad_phase_check:
                narrow_phase_response = self.narrow_phase_collision_detection(
                    body_1.position, body_1.orientation, body_1.collider_type, body_1_collider,
                    body_2.position, body_2.orientation, body_2.collider_type, body_2_collider
                )

                if narrow_phase_response.collision:
                    constraint.collision     = True
                    constraint.r_1          = narrow_phase_response.r_1
                    constraint.r_2          = narrow_phase_response.r_2
                    constraint.normal       = narrow_phase_response.normal
                    constraint.penetration  = narrow_phase_response.penetration


    @ti.func
    def get_collider_info(self, collider_type, collider_idx):
        """
            Get the collider information.

            Arguments:
            ----------
                collider_type : int
                    The collider type (0: sphere, 1: box)
                collider_idx : int
                    The collider index
        """

        if collider_type == colliders.BOX:
            return self.box_colliders[collider_idx]
        elif collider_type == colliders.SPHERE:
            return self.sphere_colliders[collider_idx]
        elif collider_type == colliders.CYLINDER:
            return self.cyllinder_colliders[collider_idx]
        elif collider_type == colliders.PLANE:
            return self.plane_colliders[collider_idx]
        elif collider_type == colliders.HEIGHTFIELD:
            return self.heightfield_colliders[collider_idx]
        else:
            return None
 
    @ti.kernel
    def step(self):

        h = self.dt / self.n_substeps

        self.collect_collision_pairs()

        for _ in ti.static(range(self.n_substeps)):   
            self.update_rigid_bodies_position_and_orientation(h)
            self.solve_positions(h)
            self.update_rigid_bodies_velocities(h)
            self.solve_velocities(h)
    
    @ti.kernel
    def compute_transformations(self):
        """
            Compute the transformations.
        """
        for i in range(self.n_rigid_bodies):
            body = self.bodies_list[i]
            if body.is_fixed: continue
            self.rigid_bodies_transformations[i] = body.get_transformation_matrix()
            
    

    def set_visual_objects(self):
        
        for body in self.bodies_list:
            name = "body " + i
            collider_type = body.collider_type
            # Get the collider from the collider list
            collider = self.get_collider_info(collider_type, body.collider_idx)

            if collider_type == colliders.BOX:
                full_extents = collider.half_extents * 2.0                
                self.visualizer[name].set_object(
                                                 g.Box(full_extents.to_numpy()), 
                                                 material = g.MeshPhongMaterial(color=0xff0000))
            elif collider_type == colliders.SPHERE:
                radius = collider.radius
                self.visualizer[name].set_object( g.Sphere(radius), 
                                                  material = g.MeshPhongMaterial(color=0xff0000))
            elif collider_type == colliders.CYLINDER:
                
                radius = collider.radius
                height = collider.height

                self.visualizer[name].set_object( g.Cylinder(height, radius = radius), 
                                                  material = g.MeshPhongMaterial(color=0xff0000))
            
            elif collider_type == colliders.PLANE:

                normal = collider.normal.to_numpy()
                offset = collider.offset.to_numpy()

                # Create a checkerboard texture for the plane
                texture = g.GenericMaterial(
                    color = 0xaaaaaa,
                    vertexColors=True
                )
                
                self.visualizer[name].set_object( g.TriangularMeshGeometry(render_plane(normal, offset)), 
                                                  material = texture)

            elif collider_type == colliders.HEIGHTFIELD:
                texture = g.GenericMaterial(
                    color = 0xaaaaaa,
                    wireframe = True,
                    vertexColors=True
                )

                # TODO: render the heightfield

                # self.visualizer[name].set_object( g.TriangularMeshGeometry(render_heightfield(normal, offset)),

                return NotImplementedError


    def render_collision_bodies(self):
        """
            Update the rigid bodies on the renderer (meshCat)
        """

        for i in range(self.n_bodies):
            if self.bodies_list[i].is_fixed: continue
            name = "body " + i
            self.visualizer[name].set_transform(self.rigid_bodies_transformations[i].to_numpy())

            

if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    pairs = [
        (1, 2),
        (3, 4),
        (5, 6)
    ]
    field = ti.field(dtype=ti.u32, shape=(len(pairs), 2))

    for i in range(len(pairs)):
        field[i, 0] = pairs[i][0]
        field[i, 1] = pairs[i][1]
    
    @ti.kernel
    def print_field():
        for i in range(field.shape[0]):
            print(field[i, 0], field[i, 1])

    print_field()

    



        

    