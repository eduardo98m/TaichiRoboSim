import taichi as ti

@ti.func
def hamilton_product(
    q1: ti.types.vector(4, float),
    q2: ti.types.vector(4, float)
) -> ti.types.vector(4, float):
    """
    Arguments:
    ----------
    q1 : ti.types.vector(4, float)
        -> Quaternion 1
    q2 : ti.types.vector(4, float)
        -> Quaternion 2
    """

    return ti.Vector([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    ])

@ti.func
def inverse(
    q: ti.types.vector(4, float)
):
    """
    Calculate the inverse of a unit quaternion a.k.a. the conjugate
    """
    return ti.Vector([q[0], -q[1], -q[2], -q[3]])

    


@ti.func
def from_axis_angle(axis  : ti.types.vector(3, ti.types.f32), 
                    angle : ti.types.f32):
    """
    Create a quaternion from an axis and an angle
    """
    sin_a_2 = ti.sin(angle / 2)
    return ti.Vector([
        ti.cos(angle / 2),
        axis[0] * sin_a_2,
        axis[1] * sin_a_2,
        axis[2] * sin_a_2
    ])


@ti.func
def from_euler_angles(
    roll : ti.types.f32,
    pitch : ti.types.f32,
    yaw : ti.types.f32
):
    """
    Create a quaternion from Euler angles
    """
    return hamilton_product(
                hamilton_product(
                                from_axis_angle(ti.Vector([0.0, 0.0, 1.0]), yaw), 
                                from_axis_angle(ti.Vector([0.0, 1.0, 0.0]), pitch)),
                from_axis_angle(ti.Vector([1.0, 0.0, 0.0]), roll))

@ti.func
def to_euler_angles(
    q : ti.types.vector(4, float)
):
    """
    Create Euler angles from a quaternion
    """
    roll = ti.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = ti.asin(2*(q[0]*q[2] - q[3]*q[1]))
    yaw = ti.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return ti.Vector([roll, pitch, yaw])

@ti.func
def to_rotation_matrix(
    q : ti.types.vector(4, float)
):
    """
        Create a rotation matrix from a unit quaternion
    """

    return ti.Matrix([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[3]*q[0]), 2*(q[1]*q[3] + q[2]*q[0])],
        [2*(q[1]*q[2] + q[3]*q[0]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[1]*q[0])],
        [2*(q[1]*q[3] - q[2]*q[0]), 2*(q[2]*q[3] + q[1]*q[0]), 1 - 2*(q[1]**2 + q[2]**2)]
    ])


@ti.func
def rotate_vector(
    q: ti.types.vector(4, float),
    v: ti.types.vector(3, float)
):
    """
    Arguments:
    ----------
    q : ti.types.vector(4, float)
        -> Quaternion
    v : ti.types.vector(3, float)
        -> Vector to be rotated
    """
    # Calculate q'
    # q' is the conjugate of q
    q_prime = inverse(q)

    
    v_rot =  hamilton_product(
        hamilton_product(q, ti.Vector([0.0, v[0], v[1], v[2]])), 
        q_prime)
    
    return ti.Vector([v_rot[1], v_rot[2], v_rot[3]])





if __name__ == "__main__":

    ti.init(ti.cpu)

    pi = 3.14159265358979323846264338327950288419716939937510


    print("Testing creation from euler anglee")
    @ti.kernel
    def test_euler_angles():
        a = from_euler_angles(0, 0, 0)
        print(a)
        b = from_euler_angles(0, 0, pi)
        print(b)

    
    test_euler_angles()

    print("Testing rotation")
    @ti.kernel
    def test_rotation():
        q = from_euler_angles(0.0, 0, -pi/2)
        v = ti.Vector([1.0, 0.0, 0.0])
        print(rotate_vector(q, v))

    test_rotation()
    
    