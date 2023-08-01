"""
    Author: Eduardo I. Lopez H.
    Date: 31/07/2023

"""

import taichi as ti
import taichi.math as tm
from quaternion import quaternion
from bodies import RigidBody
EPSILON = 1e-50

@ti.dataclass
class Constraint:
    """
        Constraint class

        Attributes:
        ----------
        lagrange_mult : ti.types.f32
            -> Lagrange multiplier
        compliance : ti.types.f32
            -> Constraint compliance
        r1 : ti.types.vector(3, ti.types.f32)
            -> Position of the constraint in object 1
        r2 : ti.types.vector(3, ti.types.f32)
            -> Position of the constraint in object 2
        direction : ti.types.vector(3, ti.types.f32)
            -> Direction of the constraint
        magnitude : ti.types.f32
            -> Magnitude of the constraint
        
    """

    lagrange_mult   : ti.types.f32
    compliance      : ti.types.f32
    r1              : ti.types.vector(3, ti.types.f32)
    r2              : ti.types.vector(3, ti.types.f32)
    direction       : ti.types.vector(3, ti.types.f32) 
    magnitude       : ti.types.f32
    constraint_type : ti.types.u8


@ti.dataclass
class ConstraintResponse:
    force               : ti.types.vector(3, ti.types.f32)
    torque              : ti.types.vector(3, ti.types.f32)  
    new_lagrange_mult   : ti.types.f32
    new_position_1      : ti.types.vector(3, ti.types.f32)
    new_orientation_1   : ti.types.vector(4, ti.types.f32)
    new_position_2      : ti.types.vector(3, ti.types.f32)
    new_orientation_2   : ti.types.vector(4, ti.types.f32)




@ti.func
def positional_constraint_lagrange_multiplier_update(
    c             : ti.types.f32,
    w_1           : ti.types.f32,
    w_2           : ti.types.f32,
    lagrange_mult : ti.types.f32,
    h             : ti.types.f32,
    alpha         : ti.types.f32
): 
    
    """
        Arguments:
        ----------
        `c` : ti.types.f32
            -> Constraint value
        w_1 : ti.types.f32
            -> Generalized inverse mass of object 1
        w_2 : ti.types.f32
            -> Generalized inverse mass of object 2
        lambda_ : ti.types.f32
            -> Lagrange multiplier
        h : ti.types.f32
            -> Time step
        alpha: ti.types.f32
            -> Contraint compliance 
    """
    alpha_p = alpha / h**2
    return  -c - alpha_p * lagrange_mult / (w_1 + w_2 + alpha_p)



@ti.func
def positional_contraint_generalized_inverse_mass(
                                                  m     : ti.types.f32,
                                                  I_inv : ti.types.matrix(3,3, ti.types.f32),
                                                  r     : ti.types.vector(3, ti.types.f32),
                                                  n     : ti.types.vector(3, ti.types.f32)
                                                  ):
    """
        Arguments:
        ----------
        m : ti.types.f32
            -> Mass
        I_inv : ti.types.matrix(3,3, ti.types.f32)
            -> Inverse inertia matrix
        r : ti.types.vector(3, ti.types.f32)
            -> Vector relative to the center of mass of the object, where the constraint is applied.
        n : ti.types.vector(3, ti.types.f32)
            -> Normal vector of the constraint
    
    """
    return 1 / m + tm.cross(r, n).transpose() @ I_inv @ tm.cross(r, n)

@ti.func
def compute_positional_constraint(
    obj_1     : RigidBody,
    r1_lc     : ti.types.vector(3, ti.types.f32),
    obj_2     : RigidBody,
    r2_lc     : ti.types.vector(3, ti.types.f32),
    h         : ti.types.f32 ,
    direction : ti.types.vector(3, ti.types.f32),
    magnitude : ti.types.f32,
    lagrange_mult : ti.types.f32,
    compliance : ti.types.f32
) -> ConstraintResponse:
    """
        Arguments:
        ----------
        obj_1 : RigidBody
            -> Object 1
        r1 : ti.types.vector(3, ti.types.f32)
            -> Vector relative to the center of mass of object 1, where the constraint is applied.
        obj_2 : RigidBody
            -> Object 2  
        r2 : ti.types.vector(3, ti.types.f32)
            -> Vector relative to the center of mass of object 2, where the constraint is applied.
        h : ti.types.f32
            -> Time step (Substep)
        direction : ti.types.vector(3, ti.types.f32)
            -> Direction of the constraint
        magnitude : ti.types.f32
            -> Magnitude of the constraint
        lagrange_mult : ti.types.f32
            -> Lagrange multiplier of the constraint   
        compliance : ti.types.f32
            -> Constraint compliance (refered as alpha in the paper of Dr. Muller)      
    """
    n = direction
    c = magnitude
    if c <= EPSILON:
        # We need to rotate the r vectors to the world frame
        r1 = quaternion.rotate_vector(obj_1.orientation, r1_lc)
        r2 = quaternion.rotate_vector(obj_2.orientation, r2_lc)

        # Calculate the generalized inverse mass
        w_1 = positional_contraint_generalized_inverse_mass(obj_1.mass, obj_1.dynamic_inv_interia, r1, n)
        w_2 = positional_contraint_generalized_inverse_mass(obj_2.mass, obj_2.dynamic_inv_interia, r2, n)

        # Calculate the Lagrange multiplier update
        new_lagrange_mult = lagrange_mult + positional_constraint_lagrange_multiplier_update(c, w_1, w_2, lagrange_mult, h, compliance)
    
    else:
        
        new_lagrange_mult = lagrange_mult

    impulse = new_lagrange_mult * n

    # Compute the position and orientation 
    if not obj_1.fixed:
        new_position_1   = obj_1.position + impulse * w_1
        new_orietation_1 = tm.normalize(obj_1.orientation + quaternion.rotate_vector(obj_1.orientation, 
                                                  0.5* obj_1.dynamic_inv_interia @ tm.cross(r1, impulse))) 
    else:
        new_position_1   = obj_1.position
        new_orietation_1 = obj_1.orientation
    
    if not obj_2.fixed:
        new_position_2   = obj_2.position - impulse * w_2
        new_orietation_2 = tm.normalize(obj_2.orientation - quaternion.rotate_vector(obj_2.orientation,
                                                        0.5* obj_2.dynamic_inv_interia @ tm.cross(r2, impulse)))
    else:
        new_position_2   = obj_2.position 
        new_orietation_2 = obj_2.orientation
    
    force = impulse / h**2

    return ConstraintResponse(
        force               = force,
        new_lagrange_mult   = new_lagrange_mult,
        new_position_1      = new_position_1,
        new_orientation_1   = new_orietation_1,
        new_position_2      = new_position_2,
        new_orientation_2   = new_orietation_2  
    )



@ti.func
def angular_constraint_lagrange_multiplier_update(
    c             : ti.types.f32,
    w_1           : ti.types.f32,
    w_2           : ti.types.f32,
    lagrange_mult : ti.types.f32,
    h             : ti.types.f32,
    alpha         : ti.types.f32
): 
    
    """
        Arguments:
        ----------
        `c`   : ti.types.f32
            -> Constraint value
        `w_1` : ti.types.f32
            -> Generalized inverse mass of object 1
        `w_2` : ti.types.f32
            -> Generalized inverse mass of object 2
        `lagrange_mult` : ti.types.f32
            -> Lagrange multiplier
        `h`   : ti.types.f32
            -> Time step
        `alpha`: ti.types.f32
            -> Contraint compliance 
    """
    alpha_p = alpha / h**2
    return  -c - alpha_p * lagrange_mult / (w_1 + w_2 + alpha_p)


@ti.func
def angular_contraint_generalized_inverse_mass( 
                                                I_inv : ti.types.matrix(3,3, ti.types.f32),
                                                n     : ti.types.vector(3, ti.types.f32)
                                               ):
    """
        Arguments:
        ----------
        I_inv : ti.types.matrix(3,3, ti.types.f32)
            -> Inverse inertia matrix
        n : ti.types.vector(3, ti.types.f32)
            -> Normal vector of the constraint
    """
    return n @ I_inv @ n

@ti.func
def compute_angular_constraint(
    obj_1         : RigidBody,
    obj_2         : RigidBody,
    h             : ti.types.f32 ,
    direction     : ti.types.vector(3, ti.types.f32),
    magnitude     : ti.types.f32,
    lagrange_mult : ti.types.f32,
) -> ConstraintResponse:
    """
    
        Arguments:
        ----------
        obj_1 : RigidBody
            -> Object 1
        obj_2 : RigidBody
            -> Object 2
        h : ti.types.f32
            -> Time step (Substep)
        direction : ti.types.vector(3, ti.types.f32)
            -> Direction of the constraint
        magnitude : ti.types.f32
            -> Magnitude of the constraint
        lagrange_mult : ti.types.f32
            -> Lagrange multiplier of the constraint    
    """
    
    n     = direction
    theta = magnitude

    if theta <= EPSILON:
        # Calculate the generalized inverse mass
        w_1 = angular_contraint_generalized_inverse_mass(obj_1.dynamic_inv_interia, n)
        w_2 = angular_contraint_generalized_inverse_mass(obj_2.dynamic_inv_interia, n)

        # Calculate the Lagrange multiplier update
        new_lagrange_mult = lagrange_mult + angular_constraint_lagrange_multiplier_update(theta, w_1, w_2, lagrange_mult, h, obj_1.alpha)
    
    else:
        new_lagrange_mult = lagrange_mult

    # claculate the angular impulse
    impulse = new_lagrange_mult * n

    # update the orientation
    if not obj_1.fixed:
        new_orientation_1 = tm.normalize(obj_1.orientation + quaternion.rotate_vector(obj_1.orientation, 0.5 * obj_1.dynamic_inv_interia @ impulse))
    else:
        new_orientation_1 = obj_1.orientation
    
    if not obj_2.fixed:
        new_orientation_2 = tm.normalize(obj_2.orientation - quaternion.rotate_vector(obj_2.orientation, 0.5 * obj_2.dynamic_inv_interia @ impulse))
    else:
        new_orientation_2 = obj_2.orientation

    torque = impulse/ h**2

    return ConstraintResponse(
        torque              = torque,
        new_lagrange_mult   = new_lagrange_mult,
        new_orientation_1   = new_orientation_1,
        new_orientation_2   = new_orientation_2
    )




if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    # Create a vector
    v = ti.Vector([1,2,3])

    # Create a matrix
    m = ti.Matrix([[1,2,3], [4,5,6], [7,8,9]])

    @ti.kernel
    def test():
        print(
            v.outer_product(m @ v)
        )

    test()







    
