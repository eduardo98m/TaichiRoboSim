import taichi.math as tm
import taichi as ti

from src.physics.quaternion import quaternion

from .colliders import BoxCollider
from .collision import CollisionResponse
from .collision_utils import get_boxes_axes, \
                                get_box_vertices, \
                                get_vertices_projection_max_and_min, \
                                get_projections_overlap


EPSILON = 1e-6
@ti.func
def box_v_box(
    box1: BoxCollider,
    box2: BoxCollider,
    position1: ti.types.vector(3, float),
    position2: ti.types.vector(3, float),
    orientation_1: ti.types.vector(4, float),
    orientation_2: ti.types.vector(4, float)
) -> CollisionResponse:
    """
        Calculates the collision response between to box colliders.
    
    """
    response = CollisionResponse(False)
    
    # Get the face normals of each box
    axes       = get_boxes_axes(orientation_1, orientation_2)
    vertices_1 = get_box_vertices(box1.half_extents, position1, orientation_1 )
    vertices_2 = get_box_vertices(box2.half_extents, position2, orientation_2 )
    
    minOverlap = float('inf')
    direction  = 0.0
    collision  = False
    normal     = ti.Vector([0.0, 0.0, 0.0], float)
    
    # Test each axis
    for i in range(axes.n):
        axis = axes[i, :]
        if i >= 6:
            axis = axis.normalized()
            # Skip the axis if it is too small 
            # TODO: Is this necessary?
            if tm.length(axis) < EPSILON:
                continue
        
        # Project the boxes onto the axis
        projection_1_min, projection_1_max = get_vertices_projection_max_and_min(vertices_1, axis)
        projection_2_min, projection_2_max = get_vertices_projection_max_and_min(vertices_2, axis)
        
        # Check if the projections overlap
        overlap, dir = get_projections_overlap(projection_1_min, 
                             projection_1_max, 
                             projection_2_min, 
                             projection_2_max)
        if overlap <= 0:
            break
        
        elif overlap < minOverlap:
            # Update the minimum overlap and collision normal
            minOverlap      = overlap
            direction       = dir
            normal = axis
            collision = True
    
    if collision:
        penetration = minOverlap
        r_1 = quaternion.rotate_vector(orientation_1,  direction * normal) * penetration
        r_2 = quaternion.rotate_vector(orientation_2, -direction * normal) * penetration
        
        response =  CollisionResponse(
            True,
            normal,
            penetration,
            r_1,
            r_2
        )
    return response






    



   