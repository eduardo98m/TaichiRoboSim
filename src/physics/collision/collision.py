
import taichi as ti

@ti.dataclass
def CollisionResponse():
    collision      : bool #ti.types.i32
    normal         : ti.types.vector(3, float)
    penetration    : ti.types.f32
    r_1            : ti.types.vector(3, float) # Contact poitn relative to the center of body 1
    r_2            : ti.types.vector(3, float) # Contact poitn relative to the center of body 2