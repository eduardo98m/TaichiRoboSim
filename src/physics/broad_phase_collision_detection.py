"""
Broad phase collision detection
"""

import taichi as ti
from colliders import Collider
import taichi.math as tm

@ti.function
def rough_collision_check(
    position_a : ti.template(),
    position_b : ti.template(),
    collision_check_radius_a : ti.template(),
    collision_check_radius_b : ti.template()
):
    """
    Rough collision check:
    Check if two colliders are close enough to each other

    Parameters
    ----------
    position_a : ti.vector(3, float)
        Position of the first collider
    position_b : ti.vector(3, float)
        Position of the second collider
    
    collision_check_radius_a : ti.types.f64
        Collision check radius of the first collider
    collision_check_radius_b : ti.types.f64
        Collision check radius of the second collider
    """
    return tm.length(position_a - position_b) < collision_check_radius_a + collision_check_radius_b


@ti.func
def broad_phase_collision_detection(
    colliders : ti.template(), 
    num_colliders : ti.template(), 
    collision_check_radius : ti.template(), 
    collision_pairs : ti.template(), 
    num_collision_pairs : ti.template()):
    """
    Broad phase collision detection
    """
    for i in range(num_colliders[None]):
        for j in range(i + 1, num_colliders[None]):
            if (colliders[i].position - colliders[j].position).norm() < collision_check_radius[None]:
                collision_pairs[num_collision_pairs[None], 0] = i
                collision_pairs[num_collision_pairs[None], 1] = j
                num_collision_pairs[None] += 1
                if num_collision_pairs[None] >= collision_pairs.shape[0]:
                    return
    return