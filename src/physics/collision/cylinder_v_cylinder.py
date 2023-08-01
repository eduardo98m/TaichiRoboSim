
import taichi as ti
import taichi.math as tm

from quaternion import quaternion

from .colliders import  CylinderCollider
from .collision import CollisionResponse

from .collision_utils import get_projections_overlap,\
                                project_cylinder
                                

EPSILON = 1e-6
@ti.func
def cylinder_v_cylinder(
    cylinder_1: CylinderCollider,
    cylinder_2: CylinderCollider,
    position_1: ti.types.vector(3, float),
    position_2: ti.types.vector(3, float),
    orientation_1: ti.types.vector(4, float),
    orientation_2: ti.types.vector(4, float)
) -> CollisionResponse:
    """
        Calculates the collision response between a box collider and a cylinder collider.

        Arguments:
        ----------
        `cylinder_1`: CylinderCollider
            -> The first cylinder collider.
        `cylinder_2`: CylinderCollider
            -> The second cylinder collider.
        `position_1`: ti.types.vector(3, float)
            -> The first cylinder's collider position.
        `position_2`: ti.types.vector(3, float)
            -> The second cylinder's collider position.
        `orientation_1`: ti.types.vector(4, float)
            -> The first cylinder's collider orientation. Given as a unit quaternion.
        `orientation_2`: ti.types.vector(4, float)
            -> The second cylinder's collider orientation. Given as a unit quaternion.
        
        Returns:
        --------
        `CollisionResponse`
            -> The collision response between the two colliders.
    """
    

    axis_1 = quaternion.rotate_vector(orientation_1, ti.Vector([0.0, 0.0, 1.0]))
        
    # Add the cylinder's axis as an additional axis
    axis_2 = quaternion.rotate_vector(orientation_2, ti.Vector([0.0, 0.0, 1.0]))

    # NOTE: I don't think we need to add the cross products of the cylinder's axes as additional axes
    # edge1a = quaternion.rotate_vector(orientation_1, ti.Vector([1,0,0]))
    # edge1b = quaternion.rotate_vector(orientation_1, ti.Vector([0,1,0]))
    # edge2a = quaternion.rotate_vector(orientation_2, ti.Vector([1,0,0])) 
    # edge2b = quaternion.rotate_vector(orientation_2, ti.Vector([0,1,0]))
    
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 4)
    
    axes[0, :] = axis_1
    axes[1, :] = axis_2
    # Add the cross product of the cylinders axes as additional axes
    axes[2, :] = tm.cross(axis_1, axis_2).normalized()
    # Add the vectors from the center of cylinder 1 to the center of cylinder 2 as additional axes
    axes[3, :] = (position_2 - position_1).normalized()

    # axes[4, :] = axis_1
    # axes[5, :] = axis_2

    # axes[6, :] = tm.cross(edge1b, edge2b).normalized()
    # axes[7, :] = tm.cross(edge1b, edge2a).normalized()
    # axes[8, :] = tm.cross(edge1a, edge2b).normalized()
    # axes[9, :] = tm.cross(edge1a, edge2a).normalized()
    
    # Test each axis
    minOverlap = float('inf')
    direction = 0.0
    
    # We also calculate the cyllinder top and bottom vertices
    top_center_1    = position_1 + axis_1 * (cylinder_1.height / 2)
    bottom_center_1 = position_1 - axis_1 * (cylinder_1.height / 2)

    top_center_2    = position_2 + axis_2 * (cylinder_2.height / 2)
    bottom_center_2 = position_2 - axis_2 * (cylinder_2.height / 2)
    
    for i in range(axes.n):
        axis = axes[i]
        if i == 2: 
            # Skip the axis if it is too small
            # TODO: Is this necessary?
            if tm.length(axis) < EPSILON:
                continue
        
        # Project the box onto the axis
        projection_1_min, projection_1_max = project_cylinder(axis_1, 
                                                                top_center_1,
                                                                bottom_center_1,
                                                                cylinder_1.radius,
                                                                axis)

        
        # Project the cylinder onto the axis
        projection_2_min, projection_2_max = project_cylinder(axis_2, 
                                                                top_center_2,
                                                                bottom_center_2,
                                                                cylinder_2.radius,
                                                                axis)

        # Check if the projections overlap
        overlap, dir = get_projections_overlap(
                                    projection_1_min,
                                    projection_1_max,
                                    projection_2_min,
                                    projection_2_max)
        if overlap <= 0:
            return CollisionResponse(False)
        
        elif overlap < minOverlap:
            # Update the minimum overlap and collision normal
            minOverlap = overlap
            direction = dir
            normal = axis
    
    # Compute the penetration depth and contact points
    penetration = minOverlap
    r_1 = quaternion.rotate_vector(orientation_1, direction * normal) * minOverlap
    r_2 = quaternion.rotate_vector(orientation_2, -direction * normal) * minOverlap
    
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_1,
        r_2
    )
                   
