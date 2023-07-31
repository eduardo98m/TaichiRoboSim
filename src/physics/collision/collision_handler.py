"""
    Collision Handler Module.

    Author: Eduardo I. Lopez H.
    Organization: GIA-USB
"""

import taichi as ti
import taichi.math as tm
from colliders import BoxCollider, SphereCollider, CylinderCollider, PlaneCollider
from collision import CollisionResponse
from typing import List, Union
from src.physics.bodies import RigidBody
from aabb_v_aabb import aabb_v_aabb
from aabb_v_plane import aabb_v_plane
from colliders import BOX, CYLINDER, SPHERE ,PLANE

from box_v_box import box_v_box
from sphere_v_sphere import sphere_v_sphere
from cylinder_v_cylinder import cylinder_v_cylinder

from sphere_v_cylinder import sphere_v_cylinder
from sphere_v_box import sphere_v_box
from box_v_cylinder import box_v_cylinder

from box_v_plane import box_v_plane
from sphere_v_plane import sphere_v_plane
from cylinder_v_plane import cylinder_v_plane


@ti.func
def broad_phase_collision_detection(body_1 : RigidBody,
                                    body_2 : RigidBody,
                                    body_1_collider: Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider],
                                    body_2_collider: Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider],
                                    aabb_1_security_factor : ti.types.f32,
                                    aabb_2_security_factor : ti.types.f32)-> bool:
    """
        Broad Phase Collision Detection Function.
        This function is used to detect possible collisions between two rigid bodies.

        Arguments:
        ----------
        `body_1` : RigidBody
            -> First rigid body.
        `body_2` : RigidBody
            -> Second rigid body.
        `body_1_collider` : Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider]
            -> Collider of the first rigid body.
        `body_2_collider` : Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider]
            -> Collider of the second rigid body.
        `aabb_1_security_factor` : ti.types.f32
            -> Security factor of the first rigid body.
        `aabb_2_security_factor` : ti.types.f32
            -> Security factor of the second rigid body.
    """
    if body_1_collider.collider_type == PLANE:
        body_2_collider.compute_aabb(body_2.position, body_2.orientation)
        return aabb_v_plane(
            body_2_collider.aabb * aabb_2_security_factor,
            body_1_collider
        )
    elif body_2_collider.collider_type == PLANE:
        body_1_collider.compute_aabb(body_1.position, body_1.orientation)
        return aabb_v_plane(
            body_1_collider.aabb * aabb_1_security_factor,
            body_2_collider
        )
    else:
        body_1_collider.compute_aabb(body_1.position, body_1.orientation)
        body_2_collider.compute_aabb(body_2.position, body_2.orientation)
        return aabb_v_aabb(
            body_1_collider.aabb * aabb_1_security_factor,
            body_2_collider.aabb * aabb_2_security_factor
        )
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
                                                  body_1_collider : Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider], 
                                                  body_2_collider : Union[BoxCollider, SphereCollider, CylinderCollider, PlaneCollider]
                                                  )-> CollisionResponse:
    """
        Narrow Phase Collision Detection and Response Function.
    """
    if body_1_collider.type == SPHERE and body_2_collider.type == SPHERE:
        return sphere_v_sphere(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == BOX and body_2_collider.type == BOX:
        return box_v_box(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == CYLINDER:
        return cylinder_v_cylinder(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == BOX:
        return sphere_v_box(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == BOX and body_2_collider.type == SPHERE:
        return flip_response(
                sphere_v_box(
                body_2_collider,
                body_1_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == CYLINDER:
        return sphere_v_cylinder(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == SPHERE:
        return flip_response(
                sphere_v_cylinder(
                body_2_collider,
                body_1_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )
    elif body_1_collider.type == BOX and body_2_collider.type == CYLINDER:
        return box_v_cylinder(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.position,
            body_1.orientation,
            body_2.orientation
        )
    elif body_1_collider.type == CYLINDER and body_2_collider.type == BOX:
        return flip_response(
                box_v_cylinder(
                body_2_collider,
                body_1_collider,
                body_2.position,
                body_1.position,
                body_2.orientation,
                body_1.orientation
            )
        )   
    elif body_1_collider.type == BOX and body_2_collider.type == PLANE:
        return box_v_plane(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_1.orientation
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == BOX:
        return flip_response(
                box_v_plane(
                body_2_collider,
                body_1_collider,
                body_2.position,
                body_2.orientation
            )
        )
    elif body_1_collider.type == SPHERE and body_2_collider.type == PLANE:
        return sphere_v_plane(
            body_1_collider,
            body_2_collider,
            body_1.position
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == SPHERE:
        return flip_response(
                sphere_v_plane(
                body_2_collider,
                body_1_collider,
                body_2.position
            )
        ) 
    elif body_1_collider.type == CYLINDER and body_2_collider.type == PLANE:
        return cylinder_v_plane(
            body_1_collider,
            body_2_collider,
            body_1.position,
            body_2.orientation
        )
    elif body_1_collider.type == PLANE and body_2_collider.type == CYLINDER:
        return flip_response(
                cylinder_v_plane(
                body_2_collider,
                body_1_collider,
                body_2.position,
                body_2.orientation
            )
        )
    else : 
        # The collision response is not handled.
        return CollisionResponse(False, ti.Vector([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0)

    



 
        
    

    

        

