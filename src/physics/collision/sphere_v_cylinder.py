
import taichi as ti
import taichi.math as tm

from quaternion import quaternion

from .collision import CollisionResponse
from .colliders import CylinderCollider, SphereCollider
from .collision_utils import  get_projections_overlap, project_cylinder



@ti.func
def sphere_v_cylinder(
    sphere          : SphereCollider,
    cylinder        : CylinderCollider,
    sphere_position : ti.types.vector(3, float),
    cylinder_position    : ti.types.vector(3, float),
    cylinder_orientation : ti.types.vector(4, float)
) -> CollisionResponse:
    """
        Calculates the collision response between a sphere collider and a cylinder collider.

        Arguments:
        ----------
        `sphere`: SphereCollider
            -> The sphere collider.
        `cylinder`: CylinderCollider
            -> The cylinder collider.
        `sphere_position`: ti.types.vector(3, float)
            -> The sphere's collider position.
        `cylinder_position`: ti.types.vector(3, float)
            -> The cylinder's collider position.
        `cylinder_orientation`: ti.types.vector(4, float)
            -> The cylinder's collider orientation. Given as a unit quaternion.
        
        Returns:
        --------
        `CollisionResponse`
            -> The collision response between the two colliders.
    """
    
    # Add the sphere's center-to-cylinder vector as an additional axis
    center_to_cylinder_axis = (cylinder_position - sphere_position).normalized()

    cylinder_axis = quaternion.rotate_vector(cylinder_orientation, ti.Vector([0.0, 0.0, 1.0]))

    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 2 )

    axes[0, :] = center_to_cylinder_axis
    axes[1, :] = cylinder_axis

    cylinder_top_center    = cylinder_position + cylinder_axis * (cylinder.height / 2)
    cylinder_bottom_center = cylinder_position - cylinder_axis * (cylinder.height / 2)
    
    # Test each axis
    minOverlap = float('inf')
    direction = 0.0
    for i in range(axes.n):
        axis = axes[i]
        
        # Project the sphere onto the axis
        # TODO : Is this correct?
        projection_sphere_min = tm.dot(sphere_position - axis * sphere.radius, axis)
        projection_sphere_max = tm.dot(sphere_position + axis * sphere.radius, axis)
        
        # Project the cylinder onto the axis
        projection_cylinder_min, projection_cylinder_max = project_cylinder(
                                                                            cylinder_axis,
                                                                            cylinder_top_center,
                                                                            cylinder_bottom_center,
                                                                            cylinder.radius,
                                                                            axis
                                                                        )
        
        # Check if the projections overlap
        overlap, dir = get_projections_overlap(projection_sphere_min,
                                  projection_sphere_max,
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
    r_sphere = quaternion.rotate_vector(cylinder_orientation, direction * normal) * minOverlap
    r_cylinder = quaternion.rotate_vector(cylinder_orientation, -direction * normal) * minOverlap
    
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_sphere,
        r_cylinder
    )

