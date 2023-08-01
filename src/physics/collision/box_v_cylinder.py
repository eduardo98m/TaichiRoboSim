import taichi as ti
import taichi.math as tm

from quaternion import quaternion

from .collision import CollisionResponse
from .colliders import BoxCollider, CylinderCollider
from .collision_utils import get_box_vertices, \
                                get_vertices_projection_max_and_min, \
                                get_projections_overlap,\
                                project_cylinder
                                

EPSILON = 1e-6
@ti.func
def box_v_cylinder(
    box: BoxCollider,
    cylinder: CylinderCollider,
    box_position: ti.types.vector(3, float),
    cylinder_position: ti.types.vector(3, float),
    box_orientation: ti.types.vector(4, float),
    cylinder_orientation: ti.types.vector(4, float)
) -> CollisionResponse:
    """
        Calculates the collision response between a box collider and a cylinder collider.

        Arguments:
        ----------
        `box` : BoxCollider
            -> The box collider.
        `cylinder` : CylinderCollider
            -> The cylinder collider.
        `box_position` : ti.types.vector(3, float)
            -> The box's collider position.
        `cylinder_position` : ti.types.vector(3, float)
            -> The cylinder's collider position.
        `box_orientation` : ti.types.vector(4, float)
            -> The box's collider orientation. Given as a unit quaternion.
        `cylinder_orientation` : ti.types.vector(4, float)
            -> The cylinder's collider orientation. Given as a unit quaternion.
        
        Returns:
        --------
        `CollisionResponse`
            -> The collision response between the two colliders.
    """
    
    # Get the box's face normals
    box_face_normals  = quaternion.to_rotation_matrix(box_orientation)
    
    # Add the cylinder's axis as an additional axis
    cylinder_axis = quaternion.rotate_vector(cylinder_orientation, ti.Vector([0.0, 0.0, 1.0]))
    
    axes = ti.Matrix.zero(7, 3, ti.f32)
    # TODO: We may need to add an additional 8th axis: (the vector from the box's center to the cylinder's center)
    axes[0:3, :] = box_face_normals
    axes[3,   :] = cylinder_axis
    
    
    
    # Add the cross products of the box's face normals and the cylinder's axis as additional axes
    for i in range(3):
        axes[4 + i,   :] = tm.cross(box_face_normals[i], cylinder_axis)
    
    # Test each axis
    minOverlap = float('inf')
    direction = 0.0
    vertices = get_box_vertices(box.half_extents, box_position, box_orientation)

    # We also calculate the cyllinder top and bottom vertices
    top_center    = cylinder_position + cylinder_axis * (cylinder.height / 2)
    bottom_center = cylinder_position - cylinder_axis * (cylinder.height / 2)
    
    for i in range(axes.n):
        axis = axes[i]

        if i >= 4: # These axes are not normalized as they come from a cross product
            axis = axis.normalized()
            # Skip the axis if it is too small
            # TODO: Is this necessary?
            if tm.length(axis) < EPSILON:
                continue
        
        # Project the box onto the axis
        
        projection_box_min, projection_box_max = get_vertices_projection_max_and_min(vertices, axis)
        
        # Project the cylinder onto the axis
        projection_cylinder_min, projection_cylinder_max = project_cylinder(cylinder_axis, 
                                                                            top_center,
                                                                            bottom_center,
                                                                            cylinder.radius,
                                                                            axis)
        
        # Check if the projections overlap
        overlap, dir = get_projections_overlap(
                                    projection_box_min,
                                    projection_box_max,
                                    projection_cylinder_min,
                                    projection_cylinder_max)
        if overlap <= 0:
            return CollisionResponse(False)
        
        elif overlap < minOverlap:
            # Update the minimum overlap and collision normal
            minOverlap = overlap
            direction = dir
            normal = axis
    
    # Compute the penetration depth and contact points
    penetration = minOverlap
    r_box      = quaternion.rotate_vector(box_orientation, direction * normal) * minOverlap
    r_cylinder = quaternion.rotate_vector(cylinder_orientation, -direction * normal) * minOverlap
    
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_box,
        r_cylinder
    )
                   
