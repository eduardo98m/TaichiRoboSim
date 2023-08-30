import taichi as ti
import taichi.math as tm
import numpy as np
from time import time, sleep
from quaternion import quaternion
from typing import Union

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from src.renderer import render_heightfield, render_plane

import src.physics.collision as collision
from src.physics.collision import BoxCollider, SphereCollider, CylinderCollider, HeightFieldCollider, PlaneCollider

from constraints import  HingeJointConstraint

from bodies.rigid_body import RigidBody
from collision_handler import broad_phase_collision_detection, narrow_phase_collision_detection_and_response
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

        self.heightfield_colliders_list      = []
        self.heightfield_x_coord_list        = []
        self.heightfield_y_coord_list        = []
        self.heightfield_data_list           = []

        self.n_bodies         = 0
        self.dt               = dt
        self.n_substeps       = n_substeps

        self.visualizer_active = use_visualizer
        if use_visualizer:
            # zmq_url = "tcp://127.0.0.1:" + visualizer_port
            # print("Visualizer running on URL: ", zmq_url)
            self.visualizer = meshcat.Visualizer()
            print(self.visualizer.url())
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
        

        self.enabled_hinge_constraints = False
        if len(self.hinge_constraints_list) > 0:
            self.hinge_constraints = HingeJointConstraint.field(shape=(len(self.hinge_constraints_list)))
            for i in range(len(self.hinge_constraints_list)):
                self.hinge_constraints[i] = self.hinge_constraints_list[i]
            self.enabled_hinge_constraints = True
        else:
            self.hinge_constraints = HingeJointConstraint.field(shape=(1))
            self.hinge_constraints[0] = HingeJointConstraint()
        
        
        #self.precompute_potential_collison_pairs()
        self.set_up_kernel()

    @ti.kernel
    def set_up_kernel(self):

        for i in range(self.n_bodies):
            self.rigid_bodies[i].compute_inv_inertia()
            self.rigid_bodies[i].update_inertia()


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
    

    def create_hinge_joint_constraint(
            self,
            body_1_idx : ti.types.i32,
            body_2_idx : ti.types.i32,
            relative_orientation : ti.types.vector(3, float),
            r_1        : ti.types.vector(3, float),
            r_2        : ti.types.vector(3, float),
            damping    : ti.types.f32,
            compliance : ti.types.f32, 
            limited    : bool = False,
            driven     : bool = False,
            drive_by_speed : bool = False, 
            lower_limit : ti.types.f32 = 0.0,
            upper_limit : ti.types.f32 = 0.0         
    ):  
        """
        
            `relative_orientation` : ti.types.vector(3, float)
                -> The relative orientation of the two bodies, in Euler angles.       
        """
        axes_1 = np.eye(3)
        roll  = relative_orientation[0]
        pitch = relative_orientation[1]
        yaw   = relative_orientation[2]
        axes_2 =  tf.rotation_matrix(roll, [1, 0, 0])[:3,:3] @ tf.rotation_matrix(pitch, [0, 1, 0])[:3,:3] @ tf.rotation_matrix(yaw, [0, 0, 1])[:3,:3]
        
        constraint = HingeJointConstraint(
            body_1_idx = body_1_idx,
            body_2_idx = body_2_idx,
            axes_1     = axes_1,
            axes_2     = axes_2,
            r_1        = r_1,
            r_2        = r_2,
            compliance = compliance,
            damping    = damping,
            limited    = limited,
            driven     = driven,
            drive_by_speed = drive_by_speed,
            lower_limit = lower_limit,
            upper_limit = upper_limit
        )

        constraint.initialize()

        self.hinge_constraints_list.append(constraint)

        
    def set_gravity_vector(self, 
                           gravity_vector : ti.types.vector(3, float)):
        self.gravity_vector = gravity_vector

        for i in range(self.n_bodies):
            self.bodies_list[i].apply_gravity(gravity_vector)

    
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
    
    def get_body(self, idx):
        return self.rigid_bodies[idx]

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
        for i in range(self.n_bodies):
            self.rigid_bodies[i].position_update(h)
    
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

        for i in range(self.n_bodies):
            self.rigid_bodies[i].velocity_update(h)
            

    # def precompute_potential_collison_pairs(self):
    #     """
    #         Precompute the collision checks.

    #         This function is called once at the beginning of the simulation.

    #         It simply looks at the rigid bodies and checks for possible collision pairs, so the 
    #         real collision detection can be done in a more efficient way.
    #     """
        
    #     for i in range(self.n_bodies):
    #         for j in range(i + 1, self.n_bodies):
    #             body_1 = self.bodies_list[i]
    #             body_2 = self.bodies_list[j]

    #             if body_1.fixed and body_2.fixed: continue

    #             #if self.collision_groups_list[body_1.collision_group : body_2.collision_group] == 0: continue

    #             self.potential_collision_pairs_list.append((i, j))
        
    #     # Create a field to store the collision pairs
    #     self.collision_pairs_constraints = CollisionPairConstraint.field(shape=(len(self.potential_collision_pairs_list)))
    #     self.broad_phase_collision_check = ti.field(dtype=ti.i32, shape=(len(self.potential_collision_pairs_list)))


    #     # Add the collision pairs to the field
    #     for i in range(len(self.potential_collision_pairs_list)):
    #         body_1_idx = self.potential_collision_pairs_list[i][0]
    #         body_2_idx = self.potential_collision_pairs_list[i][1]
    #         body_1 = self.bodies_list[body_1_idx]
    #         body_2 = self.bodies_list[body_2_idx]
            
    #         self.collision_pairs_constraints[i].body_1_idx = body_1_idx
    #         self.collision_pairs_constraints[i].body_2_idx = body_2_idx
    #         self.collision_pairs_constraints[i].init(body_1, body_2)

    @ti.kernel
    def broad_phase_collision(self):
        """
            Performs the broad phase collision detection.

            Collects the possible collision pairs and stores them in the `collision_pairs_constraints` field.

            Taken from : Section 3.5 of:
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf
        """
        
        for i in range(self.collision_pairs_constraints.shape[0]):

            constraint = self.collision_pairs_constraints[i]            
            
            body_1 = self.rigid_bodies[constraint.body_1_idx]

            body_2 = self.rigid_bodies[constraint.body_2_idx]

            broad_phase_check = broad_phase_collision_detection(body_1, 
                                                                body_2, 
                                                                self.dt)

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

            body_2 = self.rigid_bodies[constraint.body_2_idx]
            
            narrow_phase_response = narrow_phase_collision_detection_and_response(body_1,  body_2)

            if narrow_phase_response.collision:
                constraint.collision         = True
                constraint.r_1               = narrow_phase_response.r_1
                constraint.r_2               = narrow_phase_response.r_2
                constraint.contact_normal    = narrow_phase_response.normal
                constraint.penetration_depth = narrow_phase_response.penetration
            
            self.collision_pairs_constraints[i] = constraint


    @ti.kernel
    def solve_positions(self, h: ti.f32):
        """
            Applies the position correction, for each of the constraints in the 
            simulation.
        """
        # for i in range(self.collision_pairs_constraints.shape[0]):
            
        #     constraint = self.collision_pairs_constraints[i]

        #     if not constraint.collision: 
        #         constraint.reset_lambda()
        #         continue
            
        #     body_1 = self.rigid_bodies[constraint.body_1_idx]
            
        #     body_2 = self.rigid_bodies[constraint.body_2_idx]

        #     response = compute_contact_constraint(body_1, body_2, constraint, h)

        #     # Udapte the bodies and the constraint
        #     self.rigid_bodies[constraint.body_1_idx] = response.body_1
        #     self.rigid_bodies[constraint.body_2_idx] = response.body_2
        #     self.collision_pairs_constraints[i] = response.contact_constraint
        
        if self.enabled_hinge_constraints:
            for i in range(self.hinge_constraints.shape[0]):
                constraint = self.hinge_constraints[i]
                body_1 = self.rigid_bodies[constraint.body_1_idx]
                body_2 = self.rigid_bodies[constraint.body_2_idx]

                position_correction = self.hinge_constraints[i].solve_position(body_1, body_2, h)
                
                body_1.position    = position_correction.new_position_1
                body_1.orientation = position_correction.new_orientation_1
                body_2.position    = position_correction.new_position_2
                body_2.orientation = position_correction.new_orientation_2

                self.rigid_bodies[constraint.body_1_idx] = body_1
                self.rigid_bodies[constraint.body_2_idx] = body_2

    
    @ti.kernel
    def solve_velocities(self, h: ti.f32):
        """
            Applies the position correction, for each of the constraints in the 
            simulation.
        """
        # for i in range(self.collision_pairs_constraints.shape[0]):
            
        #     constraint = self.collision_pairs_constraints[i]

        #     if not constraint.collision: continue
            
        #     body_1 = self.rigid_bodies[constraint.body_1_idx]
            
        #     body_2 = self.rigid_bodies[constraint.body_2_idx]

        #     response = compute_contact_velocity_constraint(body_1, body_2, constraint, h, tm.length(self.gravity_vector))

        #     # Update the bodies and the constraint
        #     self.rigid_bodies[constraint.body_1_idx] = response.body_1
        #     self.rigid_bodies[constraint.body_2_idx] = response.body_2
    
        for i in range(self.hinge_constraints.shape[0]):
            constraint = self.hinge_constraints[i]
            body_1 = self.rigid_bodies[constraint.body_1_idx]
            body_2 = self.rigid_bodies[constraint.body_2_idx]

            correction = self.hinge_constraints[i].solve_velocity(body_1, body_2, h)
            
            body_1.orientation = correction.new_orientation_1
            body_2.orientation = correction.new_orientation_2

            self.rigid_bodies[constraint.body_1_idx] = body_1
            self.rigid_bodies[constraint.body_2_idx] = body_2


    def step(self):

        h = self.dt / self.n_substeps

        #self.broad_phase_collision()

        for _ in range(self.n_substeps): 
            #self.narrow_phase_collision()
            self.update_rigid_bodies_position_and_orientation(h)
            self.solve_positions(h)
            self.update_rigid_bodies_velocities(h)
            self.solve_velocities(h)

        print("Hinge constraint angle", self.hinge_constraints[0].current_angle * 180 / tm.pi)

        if self.visualizer_active:
            self.compute_transformations()
            self.render_collision_bodies()
    
    @ti.kernel
    def compute_transformations(self):
        """
            Compute the transformations.
        """
        for i in range(self.n_bodies):
            body = self.rigid_bodies[i]
            if body.fixed: continue
            self.rigid_bodies_transformations[i] = body.compute_transformation_matrix()
            
    

    def set_visual_objects(self):
        
        for i, body in enumerate(self.bodies_list):
            name = "body " + str(i)
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
                offset = float(collider.plane_collider.offset)

                # Create a checkerboard texture for the plane
                texture = g.GenericMaterial(
                    color = 0xaaaaaa,
                    vertexColors=True
                )
                vertices, faces = render_plane(normal, offset)
                self.visualizer[name].set_object( g.TriangularMeshGeometry(vertices, faces), 
                                                  )

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
            if self.bodies_list[i].fixed: continue
            name = "body " + str(i)
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

    



        

    