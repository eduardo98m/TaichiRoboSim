import taichi as ti
import taichi.math as tm
from colliders import SphereCollider
from quaternion import quaternion
from collision import CollisionResponse


@ti.func
def sphere_v_sphere(
    sphere1: SphereCollider,
    sphere2: SphereCollider,
    position1: ti.types.vector(3, float),
    position2: ti.types.vector(3, float),
    orientation1: ti.types.vector(4, float),
    orientation2: ti.types.vector(4, float)
) -> CollisionResponse:
    """
    Calculates the contact response between two spheres

    sphere1: SphereCollider
        First sphere collider
    
    sphere2: SphereCollider
        Second sphere collider
    
    position1: ti.types.vector(3, float)
        Position of the first sphere
    
    position2: ti.types.vector(3, float)
        Position of the second sphere
    
    orientation1: ti.types.vector(4, float)
        Orientation of the first sphere as a quaternion (It is not used in this function)
    
    orientation2: ti.types.vector(4, float)
        Orientation of the second sphere as a quaternion (It is not used in this function)
    """


    # Get the radius of the spheres
    radius1 = sphere1.radius
    radius2 = sphere2.radius

    # Calculate the relative position of the spheres
    rel_position = position2 - position1

    # Calculate the penetration between the spheres
    penetration  = radius1 + radius2 - tm.length(rel_position)

    # Check if there is a collision
    if penetration < 0:
        return CollisionResponse(False)

    # Calculate the normal of the collision
    normal = tm.normalize(rel_position)

    # Calculate the contact point
    contact_point = position1 + normal * radius1

    # Calculate the contact point relative to the center of body 1
    r_1 = contact_point - position1

    # Calculate the contact point relative to the center of body 2
    r_2 = contact_point - position2

    # Return the collision response
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_1,
        r_2
    )
    
    