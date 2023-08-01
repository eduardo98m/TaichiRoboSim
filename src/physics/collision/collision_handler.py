"""
    Collision Handler Module.

    Author: Eduardo I. Lopez H.
    Organization: GIA-USB
"""

import taichi as ti
import taichi.math as tm
from typing import List, Union

#from bodies import RigidBody
from .aabb_v_aabb import aabb_v_aabb
from .aabb_v_plane import aabb_v_plane
from .colliders import Collider
from .colliders import BOX, CYLINDER, SPHERE ,PLANE
from .collision import CollisionResponse


from .box_v_box import box_v_box
from .sphere_v_sphere import sphere_v_sphere
from .cylinder_v_cylinder import cylinder_v_cylinder

from .sphere_v_cylinder import sphere_v_cylinder
from .sphere_v_box import sphere_v_box
from .box_v_cylinder import box_v_cylinder

from .box_v_plane import box_v_plane
from .sphere_v_plane import sphere_v_plane
from .cylinder_v_plane import cylinder_v_plane


@ti.func
def broad_phase_collision_detection(body_1 : ti.template(),
                                    body_2 : ti.template(),
                                    body_1_collider: Collider,
                                    body_2_collider: Collider,
                                    aabb_safety_expansion_1 : ti.types.vector(3, float),
                                    aabb_safety_expansion_2 : ti.types.vector(3, float))-> bool:
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
def narrow_phase_collision_detection_and_response(body_1 : ti.template(), 
                                                  body_2 : ti.template(), 
                                                  body_1_collider : Collider, 
                                                  body_2_collider : Collider
                                                  )-> CollisionResponse:
    """
        Narrow Phase Collision Detection and Response Function.
    """
    response = CollisionResponse(False, ti.Vector([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0)

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

    



 
        
    

    

        

