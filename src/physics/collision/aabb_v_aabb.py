from colliders import PlaneCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm


@ti.func
def aabb_v_aabb(aabb_1 : ti.types.matrix(2,3, float) ,
                aabb_2 : ti.types.matrix(2,3, float) ) -> bool:
    """
        Calculates the collision response between two AABBs.

        Arguments:
        ----------
        `aabb_1`: ti.types.matrix(2,3, float)
            ->The first Axis aligned bounding box.
        `aabb_2`: ti.types.matrix(2,3, float)
            ->The second Axis aligned bounding box.

        Returns:
        --------
        `bool`
            ->True if the AABBs collide(overlap), false otherwise.
    """

    return (aabb_1[0][1] <= aabb_2[1][1] and
            aabb_1[1][1] >= aabb_2[0][1] and
            aabb_1[0][2] <= aabb_2[1][2] and
            aabb_1[1][2] >= aabb_2[0][2] and
            aabb_1[0][3] <= aabb_2[1][3] and
            aabb_1[1][3] >= aabb_2[0][3])