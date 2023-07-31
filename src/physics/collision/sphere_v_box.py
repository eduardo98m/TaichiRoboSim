from colliders import BoxCollider, SphereCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm
from physics.collision.collision_utils import get_box_vertices, \
                                get_vertices_projection_max_and_min, \
                                get_projections_overlap



@ti.func
def sphere_v_box(
    sphere          : SphereCollider,
    box             : BoxCollider,
    sphere_position : ti.types.vector(3, float),
    box_position    : ti.types.vector(3, float),
    box_orientation : ti.types.vector(4, float)
) -> CollisionResponse:
    
    # Get the box's face normals
    box_face_normals  = quaternion.to_rotation_matrix(box_orientation)
    
    # Add the sphere's center-to-box vector as an additional axis
    center_to_box_axis = (box_position - sphere_position).normalized()
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 4 )
    axes[0:3, :] = box_face_normals
    axes[3,   :] = center_to_box_axis
    
    # Test each axis
    minOverlap = float('inf')
    direction = 0.0
    vertices = get_box_vertices(box.half_extents, box_position, box_orientation)
    for i in range(axes.n):
        axis = axes[i]
        
        # Project the sphere onto the axis
        # TODO : Is this correct?
        projection_sphere_min = tm.dot(sphere_position - axis * sphere.radius, axis)
        projection_sphere_max = tm.dot(sphere_position + axis * sphere.radius, axis)
        
        # Project the box onto the axis
        projection_box_min, projection_box_max = get_vertices_projection_max_and_min(vertices, axis)
        
        # Check if the projections overlap
        overlap, dir = get_projections_overlap(projection_sphere_min,
                                  projection_sphere_max,
                                  projection_box_min,
                                  projection_box_max)
        if overlap <= 0:
            return CollisionResponse(False)
        
        elif overlap < minOverlap:
            # Update the minimum overlap and collision normal
            minOverlap = overlap
            direction = dir
            normal = axis
    
    # Compute the penetration depth and contact points
    penetration = minOverlap
    r_sphere = quaternion.rotate_vector(box_orientation, direction * normal) * minOverlap
    r_box = quaternion.rotate_vector(box_orientation, -direction * normal) * minOverlap
    
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_sphere,
        r_box
    )

