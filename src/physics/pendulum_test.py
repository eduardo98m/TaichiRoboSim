import taichi as ti
from physics_world import PhysicsWorld
from collision import PlaneCollider, SphereCollider, Collider, BoxCollider, CylinderCollider
from collision import PLANE, SPHERE, BOX, CYLINDER
from bodies import RigidBody, Material
from time import time, sleep
import numpy as np




def main():
    # Initiaize taichi
    ti.init(ti.cpu)
    
    dt = 1/60
    world = PhysicsWorld(dt, 20, use_visualizer = True, visualizer_port = "4343")
    # Create a plane collider
    
    radius = 0.1
    sphere_collider = Collider( type = SPHERE,
                                sphere_collider = SphereCollider(radius = radius),
                                cylinder_collider = CylinderCollider(radius = radius, height = 0.1))
    
    box_collider = Collider( type = BOX,
                                box_collider = BoxCollider(box_size = ti.Vector([0.1, 0.1, 0.1], dt = ti.f32)))
    

    # Create the plane and sphere rigid bodies
    ceramic  = Material(0.0, 1.0, 0.5, 0.5)

    sphere_1 = RigidBody(mass = 1.0,
                        inertia = ti.Matrix(np.eye(3,3, dtype = np.float32)),
                        position = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32), 
                        orientation = ti.Vector([1.0, 0.0, 0.0, 0.0], dt = ti.f32),
                        velocity = ti.Vector([0.0, 10.0, 0.0], dt = ti.f32),
                        angular_velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        fixed = True,
                        material = ceramic,
                        collider = sphere_collider)
    
    #sqrt2 = np.sqrt(2.0)
    mass = 12.0
    inertia = (2/5)* mass * radius**2 * ti.Matrix([[1, 0.0, 0.0],
                                                [0.0, 1, 0.0],
                                                [0.0, 0.0, 1]]) 
    sphere_2 = RigidBody(mass = mass,
                        inertia = inertia,
                        position = ti.Vector([0.0, -1.0, 0.0], dt=ti.f32), 
                        orientation = ti.Vector([1.0, 0.0, 0.0, 0.0], dt = ti.f32),
                        velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        angular_velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        fixed = False,
                        material = ceramic,
                        collider = sphere_collider)


    mass = 5.0
    inertia = (2/5)* mass * radius**2 * ti.Matrix([[1, 0.0, 0.0],
                                                [0.0, 1, 0.0],
                                                [0.0, 0.0, 1]]) 
    sphere_3 = RigidBody(mass = mass,
                        inertia = inertia,
                        position = ti.Vector([0.0, -1.0, 0.0], dt=ti.f32), 
                        orientation = ti.Vector([1.0, 0.0, 0.0, 0.0], dt = ti.f32),
                        velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        angular_velocity = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
                        fixed = False,
                        material = ceramic,
                        collider = sphere_collider)
    

    
    sphere_1_idx = world.add_rigid_body(sphere_1)

    sphere_2_idx = world.add_rigid_body(sphere_2)

    sphere_3_idx = world.add_rigid_body(sphere_3)



    world.set_gravity_vector(ti.Vector([0.0, 0.0, -9.81], dt = ti.f32))

    world.create_hinge_joint_constraint(
        body_1_idx= sphere_1_idx,
        body_2_idx= sphere_2_idx,
        relative_orientation = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
        r_1 = ti.Vector([ 0.0, 0.0, 0.0], dt = ti.f32),
        r_2 = ti.Vector([ 0.0, 1.0, 0.0], dt = ti.f32),
        damping = 0.0,
        compliance = 0.,
        limited= False,
        driven= False
    )

    # world.create_hinge_joint_constraint(
    #     body_1_idx= sphere_2_idx,
    #     body_2_idx= sphere_3_idx,
    #     relative_orientation = ti.Vector([0.0, 0.0, 0.0], dt = ti.f32),
    #     r_1 = ti.Vector([ 0.0, 0.0, 0.0], dt = ti.f32),
    #     r_2 = ti.Vector([ 0.0, 1.0, 0.0], dt = ti.f32),
    #     damping = 1.2,
    #     compliance = 0.,
    #     limited= False,
    #     driven= False
    #     )



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
        t_o = time() 
        
    print("Average step time: ", np.mean(step_time))
    print("Real time factor : ", dt/np.mean(step_time))



if __name__ == "__main__":
    main()