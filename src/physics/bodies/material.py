import taichi as ti

@ti.dataclass
class Material:
    """
        Class for storing material properties.
    """
    restitution_coeff : ti.types.f32
    static_friction_coeff  : ti.types.f32
    dynamic_friction_coeff : ti.types.f32
    