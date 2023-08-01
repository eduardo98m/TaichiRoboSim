import taichi as ti
from physics_world import PhysicsWorld
from collision import PlaneCollider, SphereCollider
from bodies import RigidBody, Material

def main():
    # Initiaize taichi
    ti.init(ti.cpu)

    world = PhysicsWorld(0.01, 20, use_visualizer = False, visualizer_port = "4343")

    # Create a plane collider
    plane_collider = PlaneCollider(normal = ti.Vector([0.0, 0.0, 1.0]), 
                                   offset = 0.0)
    plane_collider.init()
    spphere_collider = SphereCollider(radius = 0.5)
    spphere_collider.init()

    # Create the plane and sphere rigid bodies
    ceramic  = Material(0.0, 1.0, 0.5, 0.5)

    plane_body = RigidBody(fixed = True, material = ceramic)

    sphere = RigidBody(mass = 1.0,
                       position = ti.Vector([0.0, 0.0, 1.5], dt=ti.f32), 
                       orientation = ti.Vector([0.0, 0.0, 0.0, 1.0], dt = ti.f32),
                       velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                       angular_velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                       fixed = False,
                       material = ceramic)

    world.add_rigid_body(plane_body, plane_collider)

    world.add_rigid_body(sphere, spphere_collider)

    world.set_gravity_vector(ti.Vector([0.0, 0.0, -9.81], dt = ti.f32))

    world.set_up_simulation()

    world.step()



if __name__ == "__main__":
    main()