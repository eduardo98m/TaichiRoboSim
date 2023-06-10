from colliders import BoxCollider, PlaneCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm



@ti.func
def box_v_plane(
                box: BoxCollider,
                plane: PlaneCollider,
                box_position: ti.types.vector(3, float),
                box_orientation: ti.types.vector(4, float),
)-> CollisionResponse:
    """
        Calculates the collision response between a box and a plane.

        Arguments:
        ----------

        box: BoxCollider
            The box collider.
        
        plane: PlaneCollider
            The plane collider.
        
        box_position: ti.types.vector(3, float)
            The position of the box.
        
        box_orientation: ti.types.vector(4, float)
            The orientation of the box as a unit quaternion.
    """

    plane_normal = plane.normal
    plane_offset = plane.offset

    axis = plane_normal

    extent = tm.dot(box.half_extents, abs(quaternion.rotate_vector(box_orientation, axis)))

    distance = tm.dot(box_position, axis) + plane_offset

    if distance - extent > 0:
        return CollisionResponse(False)
    elif  distance + extent < 0:
        return CollisionResponse(False)
    else:
        return CollisionResponse(
            True,
            plane_normal,
            abs(distance) - extent,
            quaternion.rotate_vector(box_orientation, axis) * (abs(distance) - extent),
            ti.Vector([0.0, 0.0, 0.0])
        )
