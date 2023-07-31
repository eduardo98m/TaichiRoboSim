from colliders import PlaneCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm


@ti.func
def aabb_v_plane(aabb : ti.types.matrix(2,3, float) ,
                 plane : PlaneCollider) -> bool:
    """
        Calculates the collision response between an AABB and a plane.

        Arguments:
        ----------
        `aabb`: ti.types.matrix(2,3, float)
            ->The Axis aligned bounding box.
        `position`: ti.types.vector(3, float)
            ->The position of the AABB.
        `plane`: PlaneCollider
            ->The plane collider.
    """
    plane_normal = plane.normal
    plane_offset = plane.offset

    # Compute the aabb center and extents
    center = (aabb[0] + aabb[1]) * 0.5
    extents = aabb[1] - center 

    # Compute the projection interval radius of the aabb onto the plane normal
    radius = tm.dot(extents, abs(plane_normal))

    # Compute the distance of the aabb center from the plane
    distance = tm.dot(center, plane_normal) + plane_offset

    # Intersection occurs when distance falls within [-r, +r] interval
    return abs(distance) <= radius



    
