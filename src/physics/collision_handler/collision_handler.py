"""
    Collision Handler Module.

    Author: Eduardo I. Lopez H.
    Organization: GIA-USB
"""

import taichi as ti
import taichi.math as tm

from bodies import RigidBody
from collision import Collider, CollisionResponse,\
                     PLANE, SPHERE, BOX, CYLINDER,\
                     box_v_box, sphere_v_sphere, cylinder_v_cylinder,\
                     sphere_v_cylinder, sphere_v_box, box_v_cylinder,\
                     box_v_plane, sphere_v_plane, cylinder_v_plane,\
                     aabb_v_plane, aabb_v_aabb





@ti.func
def broad_phase_collision_detection(body_1 : RigidBody,
                                    body_2 : RigidBody,
                                    dt : ti.float32)-> bool:
    """
        Broad Phase Collision Detection Function.
        This function is used to detect possible collisions between two rigid bodies.

        Arguments:
        ----------
        `body_1` : RigidBody
            -> First rigid body.
        `body_2` : RigidBody
            -> Second rigid body.
        `body_1_collider` : Collider
            -> Collider of the first rigid body.
        `body_2_collider` : Collider
            -> Collider of the second rigid body.
        `aabb_1_security_factor` : ti.types.f32
            -> Security factor of the first rigid body.
        `aabb_2_security_factor` : ti.types.f32
            -> Security factor of the second rigid body.
    """
    body_1_collider = body_1.collider
    body_2_collider = body_2.collider

    aabb_safety_expansion_1 = abs(body_1.velocity) * dt * 2.0
    aabb_safety_expansion_2 = abs(body_2.velocity) * dt * 2.0


    collision  = False
    if body_1_collider.type == PLANE:
        body_2_collider.compute_aabb(body_2.position, body_2.orientation)
        expanded_aabb_2 = ti.Matrix.zero(ti.f32, 2, 3)
        expanded_aabb_2[0 , : ] = body_2_collider.aabb[0 , : ] - aabb_safety_expansion_2
        expanded_aabb_2[1 , : ] = body_2_collider.aabb[1 , : ] + aabb_safety_expansion_2
        collision =  aabb_v_plane(
            expanded_aabb_2,
            body_1_collider.plane_collider
        )
    elif body_2_collider.type  == PLANE:
        body_1_collider.compute_aabb(body_1.position, body_1.orientation)
        expanded_aabb_1 = ti.Matrix.zero(ti.f32, 2, 3)
        expanded_aabb_1[0 , : ] = body_1_collider.aabb[0 , : ] - aabb_safety_expansion_1
        expanded_aabb_1[1 , : ] = body_1_collider.aabb[1 , : ] + aabb_safety_expansion_1
        collision =  aabb_v_plane(
            expanded_aabb_1,
            body_2_collider.plane_collider
        )
    else:
        body_1_collider.compute_aabb(body_1.position, body_1.orientation)
        body_2_collider.compute_aabb(body_2.position, body_2.orientation)
        body_1_collider.compute_aabb(body_1.position, body_1.orientation)
        expanded_aabb_1 = ti.Matrix.zero(ti.f32, 2, 3)
        expanded_aabb_1[0 , : ] = body_1_collider.aabb[0 , : ] - aabb_safety_expansion_1
        expanded_aabb_1[1 , : ] = body_1_collider.aabb[1 , : ] + aabb_safety_expansion_1
        expanded_aabb_2 = ti.Matrix.zero(ti.f32, 2, 3)
        expanded_aabb_2[0 , : ] = body_2_collider.aabb[0 , : ] - aabb_safety_expansion_2
        expanded_aabb_2[1 , : ] = body_2_collider.aabb[1 , : ] + aabb_safety_expansion_2
        collision =  aabb_v_aabb(
            expanded_aabb_1,
            expanded_aabb_2
        )
    
    return collision
@ti.func
def flip_response(collision_response : CollisionResponse):
    """
        Flips the collision response.
    """
    return CollisionResponse(
            collision_response.collision,
            collision_response.normal * -1,
            collision_response.penetration,
            collision_response.r_2,
            collision_response.r_1
    )

@ti.func
def narrow_phase_collision_detection_and_response(body_1 : RigidBody, 
                                                  body_2 : RigidBody, 
                                                  )-> CollisionResponse:
    """
        Narrow Phase Collision Detection and Response Function.
    """
    response = CollisionResponse(False, ti.Vector([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0)

    body_1_collider = body_1.collider
    body_2_collider = body_2.collider

    if body_1_collider.type == SPHERE and body_2_collider.type == SPHERE:
        response = sphere_v_sphere(
            body_1_collider.sphere_collider,
            body_2_collider.sphere_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == BOX and body_2_collider.type == BOX:
        response = box_v_box(
            body_1_collider.box_collider,
            body_2_collider.box_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == CYLINDER:
        response = cylinder_v_cylinder(
            body_1_collider.cylinder_collider,
            body_2_collider.cylinder_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == BOX:
        response = sphere_v_box(
            body_1_collider.sphere_collider,
            body_2_collider.box_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == BOX and body_2_collider.type == SPHERE:
        response = flip_response(
                sphere_v_box(
                body_2_collider.sphere_collider,
                body_1_collider.box_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == CYLINDER:
        response = sphere_v_cylinder(
            body_1_collider.sphere_collider,
            body_2_collider.cylinder_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == SPHERE:
        response = flip_response(
                sphere_v_cylinder(
                body_2_collider.sphere_collider,
                body_1_collider.cylinder_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )
    elif body_1_collider.type == BOX and body_2_collider.type == CYLINDER:
        response = box_v_cylinder(
            body_1_collider.box_collider,
            body_2_collider.cylinder_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == BOX:
        response = flip_response(
                box_v_cylinder(
                body_2_collider.box_collider,
                body_1_collider.cylinder_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )   
    elif body_1_collider.type == BOX and body_2_collider.type == PLANE:
        response = box_v_plane(
            body_1_collider.box_collider,
            body_2_collider.plane_collider,
            body_1.position,
            body_1.orientation
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == BOX:
        response = flip_response(
                box_v_plane(
                body_2_collider.box_collider,
                body_1_collider.plane_collider,
                body_2.position,
                body_2.orientation
            )
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == PLANE:
        response = sphere_v_plane(
            body_1_collider.sphere_collider,
            body_2_collider.plane_collider,
            body_1.position
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == SPHERE:
        response = flip_response(
                sphere_v_plane(
                body_2_collider.sphere_collider,
                body_1_collider.plane_collider,
                body_2.position
            )
        ) 
    elif body_1_collider.type == CYLINDER and body_2_collider.type == PLANE:
        response = cylinder_v_plane(
            body_1_collider.cylinder_collider,
            body_2_collider.plane_collider,
            body_1.position,
            body_2.orientation
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == CYLINDER:
        response = flip_response(
                cylinder_v_plane(
                body_2_collider.cylinder_collider,
                body_1_collider.plane_collider,
                body_2.position,
                body_2.orientation
            )
        )
    
    return response

    



 
        
    

    

        

