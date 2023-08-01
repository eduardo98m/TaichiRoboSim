import taichi as ti


@ti.func
def aabb_v_aabb(aabb_1: ti.types.matrix(2, 3, float),
                aabb_2: ti.types.matrix(2, 3, float)) -> bool:
    """
    Calculates the collision response between two AABBs.

    Arguments:
    ----------
    `aabb_1`: ti.types.matrix(2, 3, float)
        ->The first Axis aligned bounding box.
    `aabb_2`: ti.types.matrix(2, 3, float)
        ->The second Axis aligned bounding box.

    Returns:
    --------
    `bool`
        ->True if the AABBs collide(overlap), false otherwise.
    """
    
    collision = True
    for i in range(3):
        if aabb_1[0, i] > aabb_2[1, i] or aabb_1[1, i] < aabb_2[0, i]:
            collision = False
            break
    return collision
