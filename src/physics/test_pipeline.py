import taichi as ti
from physics_world import PhysicsWorld
from collision import PlaneCollider, SphereCollider, Collider
from collision import PLANE, SPHERE
from bodies import RigidBody, Material
from time import time, sleep
import numpy as np
import quaternion

def main():
    # Initiaize taichi
    ti.init(ti.cuda)
    
    dt = 0.01
    world = PhysicsWorld(dt, 20, use_visualizer = True, visualizer_port = "4343")
    slope = ti.Vector([0,-0.08716,0.99619], dt = ti.f32)
    flat = ti.Vector([0,0,1], dt = ti.f32)
    # Create a plane collider
    plane_collider = Collider(  type = PLANE,
                                plane_collider = PlaneCollider(normal = slope, offset = 0.0))
    
    sphere_collider = Collider( type = SPHERE,
                                sphere_collider = SphereCollider(radius = 0.1))
    

    # Create the plane and sphere rigid bodies
    ceramic  = Material(0.0, 1.0, 0.5, 0.5)

    plane_body = RigidBody(fixed = True, 
                           material = ceramic, 
                           collider = plane_collider, 
                           )
    sphere = RigidBody(mass = 1.0,
                        inertia = ti.Matrix(np.eye(3,3, dtype = np.float32)),
                        position = ti.Vector([0.0, 0.0, 2.51], dt=ti.f32), 
                        orientation = ti.Vector([0.0, 0.0, 0.0, 1.0], dt = ti.f32),
                        velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        angular_velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        fixed = False,
                        material = ceramic,
                        collider = sphere_collider)

    
    sphere_idx = world.add_rigid_body(sphere)

    world.add_rigid_body(plane_body)

    world.set_gravity_vector(ti.Vector([0.0, 0.0, -9.81], dt = ti.f32))
    t_o = time()
    world.set_up_simulation()
    world.step()
    print("Compilation finished.")
    print("Compilation time: ", time() - t_o)
    step_time = []
    t_o = time()
    for _ in range(10000):
        world.step()
        step_time.append(time() - t_o)
        sleep_time = dt - step_time[-1]
        if sleep_time > 0:
            sleep(sleep_time)
        #print(world.get_body(sphere_idx).position)
        t_o = time() 
        
    print("Average step time: ", np.mean(step_time))
    print("Real time factor : ", dt/np.mean(step_time))



if __name__ == "__main__":
    main()