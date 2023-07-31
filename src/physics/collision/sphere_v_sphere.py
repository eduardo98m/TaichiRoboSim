import taichi as ti
import taichi.math as tm
from colliders import SphereCollider
from quaternion import quaternion
from collision import CollisionResponse


@ti.func
def sphere_v_sphere(
    sphere_1: SphereCollider,
    sphere_2: SphereCollider,
    position_1: ti.types.vector(3, float),
    position_2: ti.types.vector(3, float),
    orientation_1: ti.types.vector(4, float),
    orientation_2: ti.types.vector(4, float)
) -> CollisionResponse:
    """
    Calculates the contact response between two spheres

    Arguments:
    ----------

    `sphere_1`: SphereCollider
        -> First sphere collider
    
    `sphere_2`: SphereCollider
        -> Second sphere collider
    
    `position_1`: ti.types.vector(3, float)
        -> Position of the first sphere
    
    `position_2`: ti.types.vector(3, float)
        -> Position of the second sphere
    
    `orientation_1`: ti.types.vector(4, float)
        ->Orientation of the first sphere as a quaternion (It is not used in this function)
    
    `orientation_2`: ti.types.vector(4, float)
        ->Orientation of the second sphere as a quaternion (It is not used in this function)
    """


    # Get the radius of the spheres
    radius_1 = sphere_1.radius
    radius_2 = sphere_2.radius

    # Calculate the relative position of the spheres
    rel_position = position_2 - position_1

    # Calculate the penetration between the spheres
    penetration  = radius_1 + radius_2 - tm.length(rel_position)

    # Check if there is a collision
    if penetration < 0:
        return CollisionResponse(False)

    # Calculate the normal of the collision
    normal = tm.normalize(rel_position)

    # Calculate the contact point
    contact_point = position_1 + normal * radius_1

    # Calculate the contact point relative to the center of body 1
    r_1 = contact_point - position_1

    # Calculate the contact point relative to the center of body 2
    r_2 = contact_point - position_2

    # Return the collision response
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_1,
        r_2
    )
    
    