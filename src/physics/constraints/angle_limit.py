import taichi as ti
import taichi.math as tm
import quaternion

@ti.func
def angle_limit(n : ti.types.vector(3, ti.types.f32),
                n_1 : ti.types.vector(3, ti.types.f32),
                n_2 : ti.types.vector(3, ti.types.f32),
                lower_limit : ti.types.f32,
                upper_limit : ti.types.f32,
                ) :
    """
        This function calculates whether the angle between two vectors is within the limits or not
        and if not, it calculates the delta rotation that needs to be applied to satisfy the limits.

        Source : Algorithm 3 https://matthias-research.github.io/pages/publications/PBDBodies.pdf

        Note: Instead of calculating phi from the arcsin, we use the atan2 function.

        Arguments:
        ----------
        `n` : ti.types.vector(3, ti.types.f32)
            -> Common rotation axis
        `n_1` : ti.types.vector(3, ti.types.f32)
            -> Rotation axis of object 1
        `n_2` : ti.types.vector(3, ti.types.f32)
            -> Rotation axis of object 2
        `lower_limit` : ti.types.f32
            -> Lower limit
        `upper_limit` : ti.types.f32
            -> Upper limit
        
    """
    delta_q = ti.Vector.zero(ti.f32, 3)

    phi = tm.atan2(tm.dot(tm.cross(n_1, n_2), n), tm.dot(n_1, n_2))

    if phi < lower_limit or phi > upper_limit:
            
        phi = tm.clamp(phi, lower_limit, upper_limit)

        rot = quaternion.from_axis_angle(n, phi)

        n_1 = quaternion.rotate_vector(rot, n_1)

        delta_q = tm.cross(n_1, n_2)

    return False, delta_q
