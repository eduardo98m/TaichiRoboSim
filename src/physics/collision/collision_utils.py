"""
    Utility functions for the collision detection algorithms.

    Author: Eduardo I. Lopez H.
    Organization: GIA-USB
"""
import taichi as ti
import taichi.math as tm
from quaternion import quaternion

@ti.func
def get_boxes_axes(
            orientation_1: ti.types.vector(4, float), 
            orientation_2: ti.types.vector(4, float)):
    """
        Get the 15 axes from which to use the separating axes 
        theorem to calculate the collision between the two boxes.

        Arguments:
        ----------
            `orientation_1`: ti.types.vector(4, float)
                -> Orientation of the first box collider, in unit quaternion.
            `orientation_2`: ti.types.vector(4, float)
                -> Orientation of the second box collider, in unit quaternion.      
    """
    # Get the face normals of each box
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 15 )

    rot_mat_1 = quaternion.to_rotation_matrix(orientation_1)
    rot_mat_2 = quaternion.to_rotation_matrix(orientation_2)

    axes[  : 3  , : ] = rot_mat_1
    axes[3 : 6  , : ] = rot_mat_2

    axes[6 : 15 , : ] = get_boxes_edges_axes(rot_mat_1,rot_mat_2)    
    
    return axes



@ti.func
def get_vertices_projection_max_and_min(
            vertices : ti.types.matrix(8,3,float),
            axis     : ti.types.vector(3,float)
            ):
    """
        Calculates the projection of each of the box vertices onto an axis
        And returns the maximun and minimun projection, i.e. the projected points
        on the axis that are farther appart.

        Arguments:
        ---------
        `vertices`: ti.types.matrix(8,3,float)
            -> Vertices of the box collider
        `axis` : ti.types.vector(3,float)
            -> Axis from which the vertices will be projected
        
        Returns:
        --------
        `min_projection` : ti.types.vector(3,float)
            -> Minimun projected point
        `max_projection` : ti.types.vector(3,float)
            -> Maximun projected point

    """
    # Project each vertex of the box onto the axis
    min_projection =  float('inf')
    max_projection = -float('inf')

    for i in range(8):
        vertex = vertices[i, :]
        projection     = tm.dot(vertex ,axis)
        min_projection = ti.min(min_projection, projection)
        max_projection = ti.max(max_projection, projection)

    return min_projection, max_projection

@ti.func
def get_projections_overlap(
                projection_1_min : ti.types.vector(3,float), 
                projection_1_max : ti.types.vector(3,float), 
                projection_2_min : ti.types.vector(3,float), 
                projection_2_max : ti.types.vector(3,float)
                ):
    """
        Calculates the overlap given two projection edge points.

        Arguments:
        ----------
        `projection_1_min` : ti.types.vector(3,float)
            -> Minimun point of the first projection.
        `projection_1_max` : ti.types.vector(3,float)
            -> Maximun point of the first projection.
        `projection_2_min` : ti.types.vector(3,float)
            -> Minimun point of the second projection.
        `projection_2_max` : ti.types.vector(3,float)
            -> Maximun point of the second projection.
        
        Returns:
        --------
        `overlap`: ty.types.float32
            -> The overlap between the two projections
        `direction`: ty.types.float32 (This should be an int)
            -> A value of  1 if the overlap is in the direction of the firs collider
               A value of -1 if the direction of the overlap is negative to that of the firs collider.
    """
    
    overlap = ti.min(projection_1_max, projection_2_max) - ti.max(projection_1_min, projection_2_min)

    # Direction of the overlap relative to the 1st collider entity
    direction = 1.0 if projection_1_min < projection_2_min else -1.0

    return  overlap, direction
           

@ti.func
def get_box_vertices(
                half_extents : ti.types.vector(3,float), 
                position     : ti.types.vector(3,float), 
                orientation  : ti.types.vector(4,float)
                ):
    """
        Get the box 8 vertices from its orientation, position and half extents.

        Arguments:
        ----------
        `half_extents` : ti.types.vector(3,float)
            -> Half extents of the box collider
        `position` : ti.types.vector(3,float)
            -> Position of the box collider center.
        `orientation` : ti.types.vector(4,float)
            -> Orientation of the box collider as a unit quaternion.
        
        Returns:
        --------
        `vertices`: ti.types.matrix(8, 3, float)
            -> Vertices of the box collider
    """
   
    # Get the vertices of the box in local space
    x = half_extents[0]
    y = half_extents[1]
    z = half_extents[2]

    localVertices = ti.Matrix([
        [-x,-y,-z],
        [ x,-y,-z],
        [-x, y,-z],
        [ x, y,-z],
        [-x,-y, z],
        [ x,-y, z],
        [-x, y, z],
        [ x, y, z]
    ])

    vertices = ti.Matrix([ [0.0, 0.0, 0.0]  ] * 8)

    for i in range(8):
        vertices[i] = quaternion.rotate_vector(orientation, localVertices[i]) + position
    
    return vertices



@ti.func
def get_boxes_edges_axes(
                edges_1 : ti.types.matrix(3, 3, float),
                edges_2 : ti.types.matrix(3, 3, float)
                 ) -> ti.types.matrix(9, 3, float):
    """
        Calculates the axes corresponding to the cross product between the two box collider edges.
        Note that the edges of each box collider is a 3x3 matrix, this is beacause we can represent 
        the edges of the box collider with its rotation matrix.

        Note that as the edges of the box are parrallel we only need 3 of them, 
        as using the 12 edges would cause unnesesary redundancies

        Arguments:
        ---------
       `edges_1` : ti.types.matrix(3, 3, float)
            -> Rotation matrix of the first box collider
        `edges_2` : ti.types.matrix(3, 3, float)
            -> Rotation matrix of the second box collider

        Returns:
        -------
        `axes` : ti.types.matrix(9, 3, float)
            -> Cross product of each of the collider edges.
    """
    # Get the edge cross products of each box
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 9)

    for i in range(3): # 3 edges for box 1
        for j in range(3):
            axes[i+j, :] = tm.cross(edges_1[i, :], edges_2[j, :])

    return axes

@ti.func
def project_cylinder(
    cylinder_axis         : ti.types.vector(3, float),
    top_center            : ti.types.vector(3, float),
    bottom_center         : ti.types.vector(3, float),
    cylinder_radius       : float,
    axis                  : ti.types.vector(3, float)
) :
    """
        Projects a cylinder onto an axis.

        Arguments:
        ----------
            `cylinder_axis` : ti.types.vector(3, float)
                -> The cylinder's axis.
            `top_center` : ti.types.vector(3, float)
                -> The cylinder's top center.
            `bottom_center` : ti.types.vector(3, float)
                -> The cylinder's bottom center.
            `cylinder_radius` : float
                -> The cylinder's radius.
            `axis` : ti.types.vector(3, float)
                -> The axis onto which to project the cylinder.
    """
    
    
    # Project the top and bottom centers onto the axis
    projection_top    = tm.dot(top_center, axis) 
    projection_bottom = tm.dot(bottom_center, axis)
    
    # Calculate the radius of the cylinder's projection onto the axis

    projection_radius = abs(cylinder_radius * tm.length(tm.cross(cylinder_axis, axis)))
    
    # Calculate the minimum and maximum values of the cylinder's projection onto the axis
    projection_min = min(projection_top, projection_bottom) - projection_radius
    projection_max = max(projection_top, projection_bottom) + projection_radius
    
    return (projection_min, projection_max)