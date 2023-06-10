import taichi as ti
import taichi.math as tm
from colliders import SphereCollider, PlaneCollider
from quaternion import quaternion
from collision import CollisionResponse


@ti.func
def sphere_v_plane(
    sphere: SphereCollider,
    plane: PlaneCollider,
    sphere_position: ti.types.vector(3, float),
)-> CollisionResponse:
    """
        Calculates the collision response between a sphere and a plane.

        Note that the plane does not have a position or orientation. This is because the plane is infinite and it is only defined by its normal and 
        distance to the origin (the offset). The normal is a unit vector that points in the direction of the normal of the plane. 
        The distance is the distance from the origin to the plane along the normal vector.


        Currently the funtion only applies to objects above the plane (in the direction of the normal)

        Arguments: 
        ----------
        sphere: SphereCollider
            Sphere collider
        plane: PlaneCollider
            Plane collider
        sphere_position: ti.types.vector(3, float)
            Position of the sphere

    """

    # Get the radius of the sphere
    radius = sphere.radius

    # Get the normal and offset of the plane
    plane_normal = plane.normal
    plane_offset = plane.offset

    # Calculate the distance from the center of the sphere to the plane
    distance = tm.dot(sphere_position, plane_normal) + plane_offset


    if distance - radius > 0 :
        return CollisionResponse(False)
    
    elif distance + radius < 0:

        return CollisionResponse(False)
    
    else:
        penetration = radius - distance

        r_1 = -plane_normal * radius

        r_2 = tm.dot(sphere_position + r_1, plane_normal) * plane_normal

        return CollisionResponse(
            collision=True,
            normal=plane_normal,
            penetration=penetration,
            r_1=r_1,
            r_2=r_2
        )




