import taichi as ti
import taichi.math as tm
#from colliders import Collider, SphereCollider
from time import time, sleep
from quaternion import quaternion

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

colliders = {
    "sphere" : 0,
    "box"    : 1,
    "capsule": 2,
    "plane"  : 3
}


@ti.dataclass
class ColliderType:
    sphere : ti.types.u8 = 0
    box    : ti.types.u8 = 1
    plane  : ti.types.u8 = 2



@ti.dataclass
class SphereCollider():
    """
    Class for sphere colliders

    Arguments:
    ----------

    """
    radius : ti.types.f32


@ti.dataclass
class BoxCollider():
    """
    Class for box colliders

    Arguments:
    ----------

    Half extents of the box collider

    The half extents are the half of the width, height and depth of the box
    """
    half_extents : ti.types.vector(3, float)

@ti.dataclass
class PlaneCollider():
    """
    Class for plane colliders

    Arguments:
    ----------

    """
    normal : ti.types.vector(3, float)
    offset : ti.types.f32

@ti.dataclass
class Collider():
    """
    Base class for all colliders

    Arguments:
    ----------

    """
    base_collision_check_radius : ti.types.f32
    collider_type               : ti.types.u8
    sphere_data                 : SphereCollider
    box_data                    : BoxCollider
    plane_data                  : PlaneCollider


def create_box_collider(
        width  : ti.types.f32,
        height : ti.types.f32,
        depth  : ti.types.f32
):
    """
    Creates a box collider

    Arguments:
    ----------
    width : ti.types.f32
        -> Width of the box collider
    height : ti.types.f32
        -> Height of the box collider
    depth : ti.types.f32
        -> Depth of the box collider
    """
    return Collider(
        base_collision_check_radius = ti.sqrt(width**2 + height**2 + depth**2),
        collider_type               = ColliderType().box,
        box_data                    = BoxCollider(
                                    half_extents = ti.Vector([width, height, depth]))
    )

def create_sphere_collider(
        radius : ti.types.f32
):
    """
    Creates a sphere collider

    Arguments:
    ----------
    radius : ti.types.f32
        -> Radius of the sphere collider
    """
    return Collider(
        base_collision_check_radius = radius,
        collider_type               = ColliderType().sphere,
        sphere_data                 = SphereCollider(radius = radius)
    )

def create_plane_collider(
        normal : ti.types.vector(3, float),
        offset : ti.types.f32
):
    """
    Creates a plane collider

    Arguments:
    ----------
    normal : ti.types.vector(3, float)
        -> Normal of the plane collider
    offset : ti.types.f32
        -> Offset of the plane collider
    """
    return Collider(
        base_collision_check_radius = 0.0,
        collider_type               = ColliderType().plane,
        plane_data                  = PlaneCollider(
                                    normal = normal,
                                    offset = offset)
    )


@ti.dataclass
class StaticObject:
    position         : ti.types.vector(3, float)
    orientation      : ti.types.vector(4, float)
    collider         : Collider



@ti.dataclass
class DynamicObject:
    """
    Class for all objects that can move and collide with each other
    
    Arguments:
    ----------
    mass : ti.types.f32
        Mass of the object
    inertia : ti.types.matrix(3,3, float)
        Inertia tensor of the object
    position : ti.types.vector(3, float)
        Position of the object in the world
    velocity : ti.types.vector(3, float)
        Velocity of the object
    orientation : ti.types.vector(4, float)
        Orientation of the object in the world as a quaternion
    angular_velocity : ti.types.vector(3, float)
        Angular velocity of the object
    collider : Collider
        Collider of the object.
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
    collider_type       : ti.types.u8 
    collider            : Collider
    
    @ti.func
    def compute_inv_inertia(self):
        self.inv_inertia = tm.inverse(self.inertia)

    @ti.func
    def compute_dynamic_inv_inertia(self):
       rotation_matrix          = quaternion.to_rotation_matrix(self.orientation)       
       self.dynamic_inv_interia = rotation_matrix @ self.inv_inertia @ rotation_matrix.transpose()



@ti.dataclass
class CollisionData:
    collision : bool
    normal    : ti.types.vector(3, float)
    penetration_depth : ti.types.f32
    position  : ti.types.vector(3, float)

@ti.func
def compute_sphere_sphere_collision(
    sphere_1_pos    : ti.types.vector(3, float),
    sphere_1_radius : ti.types.f32,
    sphere_2_pos    : ti.types.vector(3, float),
    sphere_2_radius : ti.types.f32
) -> CollisionData:
    
    collision_data = CollisionData(
        collision         = False,
        normal            = ti.Vector([0.0, 0.0, 0.0]),
        penetration_depth = 0.0,
        position          = ti.Vector([0.0, 0.0, 0.0])
    )
    
    # Compute the distance between the two spheres

    dist = tm.length(sphere_1_pos - sphere_2_pos)
    if dist < (sphere_1_radius + sphere_2_radius):
        collision_data.collision         = True
        collision_data.normal            = tm.normalize(sphere_1_pos - sphere_2_pos)
        collision_data.penetration_depth = sphere_1_radius + sphere_2_radius - dist
        collision_data.position          = sphere_1_pos + collision_data.normal * sphere_1_radius
    
    return collision_data

@ti.func
def compute_sphere_box_collision(
    sphere_object : DynamicObject,
    box_object    : DynamicObject
) -> CollisionData:
    
    collision_data = CollisionData(
        collision         = False,
        normal            = ti.Vector([0.0, 0.0, 0.0]),
        penetration_depth = 0.0,
        position          = ti.Vector([0.0, 0.0, 0.0])
    )

    # Compute the position of the sphere in the box's local space
    sphere_pos_in_box_space =  quaternion.rotate_vector(
                                                quaternion.inverse(box_object.orientation)  
                                             ,(sphere_object.position - box_object.position)
                                             )

    # Compute the closest point on the box to the sphere
    closest_point = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if sphere_pos_in_box_space[i] < -box_object.collider.box_data.half_extents[i]:
            closest_point[i] = -box_object.collider.box_data.half_extents[i]
        elif sphere_pos_in_box_space[i] > box_object.collider.box_data.half_extents[i]:
            closest_point[i] = box_object.collider.box_data.half_extents[i]
        else:
            closest_point[i] = sphere_pos_in_box_space[i]
    
    # Compute the distance between the closest point and the sphere
    dist = tm.length(closest_point - sphere_pos_in_box_space)
    if dist < sphere_object.collider.sphere_data.radius:
        collision_data.collision = True
        collision_data.normal    = tm.normalize(sphere_pos_in_box_space - closest_point)
        collision_data.penetration_depth = sphere_object.collider.sphere_data.radius - dist
        collision_data.position  = box_object.position + quaternion.rotate_vector(box_object.orientation, closest_point)

    

@ti.data_oriented
class PhysicsWorld():
    def __init__(self, 
                 dt : ti.types.f32,
                 n_substeps: ti.types.u32) -> None:
        self.gravity_vector   = ti.Vector([0.0, -9.8, 0.0])
        self.dynamic_objects  = DynamicObject.field(shape=(10,))
        self.impulses         = ti.Vector.field(3, float, shape=(10,))
        
        self.static_objects   = StaticObject.field(shape=(10,))
        self.n_dyn_objects    = 0
        self.n_static_objects = 0
        self.n_objects        = 0
        self.dt = dt
        self.n_substeps = n_substeps
    
    @ti.kernel
    def compute_inv_inertia(self):
        for i in range(self.n_dyn_objects):
            self.dynamic_objects[i].compute_inv_inertia()
    
    def set_gravity_vector(self, 
                           gravity_vector : ti.types.vector(3, float)):
        self.gravity_vector = gravity_vector
    
    
    def add_static_object(self, 
                            object : StaticObject):
        self.static_objects[self.n_static_objects] = object
        self.n_static_objects += 1
        self.n_objects += 1

    def add_dynamic_object(self,
                            dyn_object : DynamicObject):
        self.dynamic_objects[self.n_dyn_objects] = dyn_object
        self.n_dyn_objects += 1
        self.n_objects     += 1

    @ti.func
    def integrate_dynamic_bodies(
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
        for i in range(self.n_dyn_objects):
            obj = self.dynamic_objects[i]
            # Update velocity

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
    def update_dynamic_bodies_velocities(
        self,
        h : ti.types.f32
    ):
        """

            Taken from : Algorithm 2, pag 5.
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf

            Arguments:
            ----------
                h : float 
                    The time step (substep)     
        """

        for i in range(self.n_dyn_objects):

            obj = self.dynamic_objects[i]
            obj.velocity = (obj.position - obj.previous_position) / ti.static(h)

            delta_orientation = quaternion.hamilton_product(obj.orientation, 
                                            quaternion.inverse(obj.previous_orientation))
            

            angular_velocity  = 2.0 * ti.Vector([
                                                    delta_orientation[1],
                                                    delta_orientation[2],
                                                    delta_orientation[3]
                                                ])/  ti.static(h)

            if delta_orientation[0] >=  0.0: 
                obj.angular_velocity = - angular_velocity
            else:
                obj.angular_velocity = angular_velocity
    
    @ti.func
    def solve_dynamic_bodies_position(
        self
    ):
        """
            Solve the position constraint using the Baumgarte stabilization technique.

            Taken from : Algorithm 3, pag 6.
                https://matthias-research.github.io/pages/publications/PBDBodies.pdf
        """
        for i in range(self.n_dyn_objects):
            obj = self.dynamic_objects[i]
            # Update position
            obj.position = obj.position + obj.position_correction

            # Update orientation
            obj.orientation = tm.normalize(
                obj.orientation + 0.5 * obj.orientation_correction
            )


    
    @ti.func
    def compute_collisions(
        self
    ):
        # Set the impulses to zero
        self.impulses.fill(ti.Vector([0.0, 0.0, 0.0]))
        for i in range(
             self.n_dyn_objects
        ):
            
            for j in range(
                i + 1, self.n_dyn_objects
            ):
                # Get the collider type
                collider_type_j = self.dynamic_objects[j].collider_type
                collider_type_i = self.dynamic_objects[i].collider_type

                
                if collider_type_j == ColliderType().sphere and collider_type_i == ColliderType().sphere: 
                    sphere_1_pos = self.dynamic_objects[i].position
                    sphere_1_radius = self.dynamic_objects[i].collider.sphere_data.radius
                    sphere_2_pos = self.dynamic_objects[j].position
                    sphere_2_radius = self.dynamic_objects[j].collider.sphere_data.radius
 
                    collision_data = compute_sphere_sphere_collision(
                        sphere_1_pos,
                        sphere_1_radius,
                        sphere_2_pos,
                        sphere_2_radius
                    )

                    if collision_data.collision:
                        self.impulses[i] =  collision_data.normal  * collision_data.penetration_depth * 1000
                        self.impulses[j] = -collision_data.normal  * collision_data.penetration_depth * 1000                  
    @ti.kernel
    def step(self):

        h = self.dt / self.n_substeps

        for i in ti.static(range(self.n_substeps)):   
            self.integrate_dynamic_bodies()
            self.compute_collisions()

if __name__ == "__main__":
    # Create a new visualizer
    vis = meshcat.Visualizer()
    vis.open()

    # init taichi
    ti.init(arch=ti.cpu)

    r1 = 0.1
    r2 = 0.3
    pos_o = ti.Vector([1.0, 0.0, 0.0])
    vel_o = ti.Vector([1.0, 0.0, 0.0]) * 2.0
    # Create a dynamic object
    sphere = DynamicObject(
        mass             = 1.0,
        inertia          = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ti.f32),
        position         = pos_o,
        velocity         = -vel_o,
        orientation      = ti.Vector([0.0, 0.0, 0.0, 1.0]),
        angular_velocity = ti.Vector([0.0, 0.0, 0.0]),
        collider         = Collider(
            base_collision_check_radius = 1.0,
            collider_type               = ColliderType().sphere,
            sphere_data                 = SphereCollider(radius=r1),
        )
    )

    




    sphere_2 = DynamicObject(
        mass             = 1.0,
        inertia          = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ti.f32),
        position         = -pos_o,
        velocity         = vel_o,
        orientation      = ti.Vector([0.0, 0.0, 0.0, 1.0]),
        angular_velocity = ti.Vector([0.0, 0.0, 0.0]),
        collider         = Collider(
            base_collision_check_radius = 0.05,
            collider_type               = ColliderType().sphere,
            sphere_data                 = SphereCollider(radius=r2),
        )
    )

    plane_verices = ti.Matrix([
        [-1, -1, 0],  # the first vertex is at [0, 0, 0]
        [-1, 1, 0],
        [1, 1, 0],
        [1, -1, 0]
    ]) *20

    plane_faces = ti.Matrix([
        [0, 1, 2],  # The first face consists of vertices 0, 1, and 2
        [3, 0, 2]
    ])

    print(plane_verices.to_numpy())
    print(plane_faces.to_numpy())

    vis["sphere_1"].set_object(g.Sphere(r1), material = g.MeshPhongMaterial(color=0xff0000))
    vis["sphere_2"].set_object(g.Sphere(r2), material = g.MeshPhongMaterial(color=0x00ff00, wireframe=True))
    vis["plane"].set_object(g.TriangularMeshGeometry(
        plane_verices.to_numpy(),
        plane_faces.to_numpy()
    ), material = g.MeshToonMaterial(color=0x0000ff,
                                        
                                        ))
    # Create a physics world
    world = PhysicsWorld(dt=0.002)
    world.add_dynamic_object(sphere)
    world.set_gravity_vector(ti.Vector([0.0, 0.0, 0.0]))
    world.add_dynamic_object(sphere_2)
    world.compute_inv_inertia()

    # window = ti.ui.Window("Rigid body simulation", (1280, 720),
    #                   vsync=True)
    # canvas = window.get_canvas()
    # canvas.set_background_color((1, 1, 1))
    # scene = ti.ui.Scene()
    # camera = ti.ui.Camera()


    fov = 60.

    # camera.position(0.0, 0.0, 5.0)
    # camera.lookat(0.0, 0.0, 0)
    
    dt = world.dt
    frame_rate = 1/dt
    current_t = 0.0
    # ball_1_position = ti.Vector.field(3, float, shape=1)
    # ball_2_position = ti.Vector.field(3, float, shape=1)
    input("Start!")
    while True:
        t_old = time()
        # if current_t > 1.5:
        #     # Reset
        #     world.dynamic_objects[0].position = ti.Vector([0.0, 0.0, 0.0])
        #     world.dynamic_objects[0].velocity = ti.Vector([0.0, 0.0, 0.0])
        #     world.dynamic_objects[1].position = ti.Vector([0.1, 0.1, 0.1])
        #     world.dynamic_objects[1].velocity = ti.Vector([0.0, -0.5, 0.0])
        #     current_t = 0

        
        current_t += dt
        #update_vertices()
        #update_plane_vertices()
        # Lets add camera control with mouse
        
        #camera.position(0.0, 0.0, 3)
        #camera.lookat(0.0, 0.0, 0)
        # camera.fov(fov)
        # camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        # scene.set_camera(camera)
        

        # scene.point_light(pos=camera.curr_position, color=(1, 1, 1))
        # scene.ambient_light((0.5, 0.5, 0.5))

        # ball_1_position[0] = world.dynamic_objects[0].position
        # ball_2_position[0] = world.dynamic_objects[1].position
        vis["sphere_1"].set_transform(tf.translation_matrix(world.dynamic_objects[0].position.to_list()))
        vis["sphere_2"].set_transform(tf.translation_matrix(world.dynamic_objects[1].position.to_list()))
        #scene.particles(centers = ball_1_position, radius = r1, color=(0.5, 0.42, 0.8))
        #scene.particles(centers = ball_2_position, radius = r2, color=(0.1, 0.42, 0.7))
        #canvas.scene(scene)
        world.step()
        t_frame = time() - t_old
        if t_frame < 1. / frame_rate:
            sleep(
                1. / frame_rate - t_frame
            )
        #window.show()
        #input("step")


        

    