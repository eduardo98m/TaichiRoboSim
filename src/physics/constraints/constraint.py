import taichi as ti

@ti.dataclass
class ConstraintData:
    """
        Constraint class

        Attributes:
        ----------
        `lagrange_mult` : ti.types.f32
            -> Lagrange multiplier
        `compliance` : ti.types.f32
            -> Constraint compliance
        `r_1` : ti.types.vector(3, ti.types.f32)
            -> Position of the constraint in object 1
        `r_2` : ti.types.vector(3, ti.types.f32)
            -> Position of the constraint in object 2
        `direction` : ti.types.vector(3, ti.types.f32)
            -> Direction of the constraint
        `magnitude` : ti.types.f32
            -> Magnitude of the constraint
        
    """

    lagrange_mult   : ti.types.f32
    compliance      : ti.types.f32
    r_1             : ti.types.vector(3, ti.types.f32)
    r_2             : ti.types.vector(3, ti.types.f32)
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

