from colliders import BoxCollider, CylinderCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm


@ti.func
def box_v_cylinder(collider1: BoxCollider, 
                   collider2: CylinderCollider,
                   position1: ti.types.vector(3, float),
                   position2: ti.types.vector(3, float),
                   orientation1: ti.types.vector(4, float),
                   orientation2: ti.types.vector(4, float)) -> CollisionResponse:
    """
    """
    return CollisionResponse(False)
                   
