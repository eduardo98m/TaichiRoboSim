import taichi as ti
import taichi.math as tm

from quaternion import quaternion

from .collision import CollisionResponse
from .colliders import CylinderCollider, PlaneCollider
from .collision_utils import project_cylinder

EPSILON = 1e-6

@ti.func 
def cylinder_v_plane(
    cylinder: CylinderCollider, 
    plane: PlaneCollider,
    cylinder_position: ti.types.vector(3, float),
    cylinder_orientation: ti.types.vector(4, float)
) -> CollisionResponse:
    """
        Calculates the collision response between a cylinder collider and a plane collider.

        Arguments:
        ----------
        `cylinder` : CylinderCollider
            -> The cylinder collider.
        `plane` : PlaneCollider
            -> The plane collider.
        `cylinder_position` : ti.types.vector(3, float)
            -> The cylinder's collider position.
        `cylinder_orientation` : ti.types.vector(4, float)
            -> The cylinder's collider orientation. Given as a unit quaternion.
    """
    
    plane_normal = plane.normal
    plane_offset = plane.offset
    
    cylinder_axis = quaternion.rotate_vector(cylinder_orientation, ti.Vector([0, 0, 1]))

    axis = plane_normal
    
    top_center    = cylinder_position + cylinder_axis * (cylinder.height / 2)
    bottom_center = cylinder_position - cylinder_axis * (cylinder.height / 2)
        
        
    # Project the box onto the axis
    projection_1_min, projection_1_max = project_cylinder(cylinder_axis, 
                                                            top_center,
                                                            bottom_center,
                                                            cylinder.radius,
                                                            axis)

    

    # Check if the plane offset is greater than the projection max or less than the projection min
    # If it is, then the box is not colliding with the plane
    if plane_offset > projection_1_max or plane_offset < projection_1_min:
        return CollisionResponse(False)
    
    # If the plane offset is between the projection min and max, then the box is colliding with the plane
    # Calculate the overlap
    overlap = plane_offset - projection_1_min

    
    # Update the minimum overlap and collision normal
    minOverlap = overlap
    direction = 1.0 # The direction is set to positive because we want the planes to act as if they were one-sided
    normal = axis
    
    # Compute the penetration depth and contact points
    penetration = minOverlap
    r_1 = quaternion.rotate_vector(cylinder_orientation, direction * normal) * minOverlap
    
    return CollisionResponse(
        True,
        normal,
        penetration,
        r_1,
        ti.Vector([0.0, 0.0, 0.0])
    )