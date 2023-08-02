import taichi as ti
import taichi.math as tm
from time import time, sleep
from quaternion import quaternion
from typing import Union

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from src.renderer import render_heightfield, render_plane

import src.physics.collision as collision
from src.physics.collision import BoxCollider, SphereCollider, CylinderCollider, HeightFieldCollider, PlaneCollider

from constraints import constraint_types, CollisionPairConstraint

from bodies.rigid_body import RigidBody
from collision import broad_phase_collision_detection, narrow_phase_collision_detection_and_response
from constraints import compute_contact_constraint, compute_contact_velocity_constraint



@ti.data_oriented
class PhysicsWorld():
    
    def __init__(self, 
                 dt : ti.types.f32,
                 n_substeps: ti.types.u32,
                 use_visualizer = True,
                 visualizer_port = "4343") -> None:
        
        self.gravity_vector   = ti.Vector([0.0, 0.0, -9.8])
        
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
            print("Visualizer opened")
    

    def set_up_simulation(self):
        self.rigid_bodies               = RigidBody.field(shape=(self.n_bodies,))
        
        if self.visualizer_active:
            self.rigid_bodies_transformations  = tm.mat4.field(shape=(self.n_bodies,))
            self.set_visual_objects()
        # Add the rigid bodies in the rigid bodies field
        for i in range(self.n_bodies):
            self.rigid_bodies[i] = self.bodies_list[i] 
        
        self.precompute_potential_collison_pairs()

    
    def add_height_field(self, 
                         height_field  : HeightFieldCollider,
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
                       body : RigidBody
                       ) -> ti.types.u32:
        """
            Adds a rigid body to the simulation.
            Returns the index of the rigid body in the rigid bodies list.

            This function cannot be called after the simulation has started.

            Arguments:
            ----------
            * `body` : RigidBody
                -> The rigid body to be added to the simulation.
        """


        self.bodies_list.append(body)
        self.n_bodies += 1

        return self.n_bodies - 1

    @ti.kernel
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
        for i in ti.static(range(self.n_bodies)):
            obj = self.rigid_bodies[i]
            # Check if the object is static
            if not obj.fixed:
                # Save the previous state
                obj.prev_position = obj.position
                obj.prev_orientation = obj.orientation
                obj.prev_velocity = obj.velocity
                obj.prev_angular_velocity = obj.angular_velocity

                # Update velocity
                obj.velocity    = obj.velocity + (self.gravity_vector  + obj.external_force/ obj.mass) * h 
                
                # Update position
                obj.position    = obj.position + obj.velocity  * h 
                
                # Update angular velocity
                obj.angular_velocity = obj.angular_velocity +  h * obj.inv_inertia @ (
                    obj.external_torque - tm.cross(obj.angular_velocity, obj.inertia @ obj.angular_velocity)
                )
                # Update orientation
                obj.orientation = quaternion.rotate_by_axis(obj.orientation, obj.angular_velocity, h)
    @ti.kernel
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

        for i in ti.static(range(self.n_bodies)):
            obj = self.rigid_bodies[i]
            if not obj.fixed:
                obj.velocity = (obj.position - obj.prev_position) / h

                delta_orientation = quaternion.hamilton_product(obj.orientation, 
                                                quaternion.inverse(obj.prev_orientation))
                

                angular_velocity  = 2.0 * ti.Vector([   delta_orientation[1],
                                                        delta_orientation[2],
                                                        delta_orientation[3]
                                                    ])/  h

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

                #if self.collision_groups_list[body_1.collision_group : body_2.collision_group] == 0: continue

                self.potential_collision_pairs_list.append((i, j))
        
        # Create a field to store the collision pairs
        self.collision_pairs_constraints = CollisionPairConstraint.field(shape=(len(self.potential_collision_pairs_list)))
        self.broad_phase_collision_check = ti.field(dtype=ti.i32, shape=(len(self.potential_collision_pairs_list)))


        # Add the collision pairs to the field
        for i in range(len(self.potential_collision_pairs_list)):
            body_1_idx = self.potential_collision_pairs_list[i][0]
            body_2_idx = self.potential_collision_pairs_list[i][1]
            body_1 = self.bodies_list[body_1_idx]
            body_2 = self.bodies_list[body_2_idx]
            
            self.collision_pairs_constraints[i].body_1_idx = body_1_idx
            self.collision_pairs_constraints[i].body_2_idx = body_2_idx
            self.collision_pairs_constraints[i].init(body_1, body_2)

    @ti.kernel
    def broad_phase_collision(self):
        """
            Performs the broad phase collision detection.

            Collects the possible collision pairs and stores them in the `collision_pairs_constraints` field.

            Taken from : Section 3.5 of:
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf
        """
        
        for i in ti.static(range(self.collision_pairs_constraints.shape[0])):

            constraint = self.collision_pairs_constraints[i]            
            
            body_1 = self.rigid_bodies[constraint.body_1_idx]
            
            body_1_collider = body_1.collider

            body_2 = self.rigid_bodies[constraint.body_2_idx]

            body_2_collider = body_2.collider

            aabb_safety_expansion_1 = abs(body_1.velocity) * self.dt * 2.0
            aabb_safety_expansion_2 = abs(body_2.velocity) * self.dt * 2.0

            broad_phase_check = broad_phase_collision_detection(body_1, 
                                                                body_2, 
                                                                body_1_collider, 
                                                                body_2_collider, 
                                                                aabb_safety_expansion_1, 
                                                                aabb_safety_expansion_2
            )

            self.broad_phase_collision_check[i] = broad_phase_check


    @ti.kernel
    def narrow_phase_collision(self):
        """
            Collect the collision pairs.
        """
        
        for i in range(self.collision_pairs_constraints.shape[0]):
            
            broad_phase_check = self.broad_phase_collision_check[i]

            if not broad_phase_check: continue
                
            constraint = self.collision_pairs_constraints[i]

            body_1 = self.rigid_bodies[constraint.body_1_idx]
            
            body_1_collider = body_1.collider

            body_2 = self.rigid_bodies[constraint.body_2_idx]

            body_2_collider = body_2.collider
            
            narrow_phase_response = narrow_phase_collision_detection_and_response(body_1,
                                                                                    body_2,
                                                                                    body_1_collider,
                                                                                    body_2_collider
                                                                                )

            if narrow_phase_response.collision:
                constraint.collision         = True
                constraint.r_1               = narrow_phase_response.r_1
                constraint.r_2               = narrow_phase_response.r_2
                constraint.contact_normal    = narrow_phase_response.normal
                constraint.penetration_depth = narrow_phase_response.penetration


    @ti.kernel
    def solve_positions(self, h: ti.f32):
        """
            Applies the position correction, for each of the constraints in the 
            simulation.
        """
        for i in range(self.collision_pairs_constraints.shape[0]):
            
            constraint = self.collision_pairs_constraints[i]

            if not constraint.collision: continue
            
            body_1 = self.rigid_bodies[constraint.body_1_idx]
            
            body_2 = self.rigid_bodies[constraint.body_2_idx]

            response = compute_contact_constraint(body_1, body_2, constraint, h)

            # Udapte the bodies and the constraint
            body_1 = response.body_1
            body_2 = response.body_2
            constraint = response.contact_constraint
    
    @ti.kernel
    def solve_velocities(self, h: ti.f32):
        """
            Applies the position correction, for each of the constraints in the 
            simulation.
        """
        for i in range(self.collision_pairs_constraints.shape[0]):
            
            constraint = self.collision_pairs_constraints[i]

            if not constraint.collision: continue
            
            body_1 = self.rigid_bodies[constraint.body_1_idx]
            
            body_2 = self.rigid_bodies[constraint.body_2_idx]

            response = compute_contact_velocity_constraint(body_1, body_2, constraint, h, tm.length(self.gravity_vector))

            # Udapte the bodies and the constraint
            body_1 = response.body_1
            body_2 = response.body_2

    def step(self):

        h = self.dt / self.n_substeps

        self.broad_phase_collision()

        for _ in ti.static(range(self.n_substeps)): 
            self.narrow_phase_collision()  
            self.update_rigid_bodies_position_and_orientation(h)
            self.solve_positions(h)
            self.update_rigid_bodies_velocities(h)
            self.solve_velocities(h)
        
        if self.visualizer_active:
            self.compute_transformations()
            self.render_collision_bodies()
    
    @ti.kernel
    def compute_transformations(self):
        """
            Compute the transformations.
        """
        for i in range(self.n_bodies):
            body = self.bodies_list[i]
            if body.is_fixed: continue
            self.rigid_bodies_transformations[i] = body.get_transformation_matrix()
            
    

    def set_visual_objects(self):
        
        for body in self.bodies_list:
            name = "body " + i
            collider_type = body.collider.type
            # Get the collider from the collider list
            collider = body.collider

            if collider_type == collision.BOX:
                full_extents = collider.box_collider.half_extents * 2.0                
                self.visualizer[name].set_object(
                                                 g.Box(full_extents.to_numpy()), 
                                                 material = g.MeshPhongMaterial(color=0xff0000))
            elif collider_type == collision.SPHERE:
                radius = collider.sphere_collider.radius
                self.visualizer[name].set_object( g.Sphere(radius), 
                                                  material = g.MeshPhongMaterial(color=0xff0000))
            elif collider_type == collision.CYLINDER:
                
                radius = collider.cylinder_collider.radius
                height = collider.cylinder_collider.height

                self.visualizer[name].set_object( g.Cylinder(height, radius = radius), 
                                                  material = g.MeshPhongMaterial(color=0xff0000))
            
            elif collider_type == collision.PLANE:

                normal = collider.plane_collider.normal.to_numpy()
                offset = collider.plane_collider.offset.to_numpy()

                # Create a checkerboard texture for the plane
                texture = g.GenericMaterial(
                    color = 0xaaaaaa,
                    vertexColors=True
                )
                
                self.visualizer[name].set_object( g.TriangularMeshGeometry(render_plane(normal, offset)), 
                                                  material = texture)

            elif collider_type == collision.HEIGHTFIELD:
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

    



        

    