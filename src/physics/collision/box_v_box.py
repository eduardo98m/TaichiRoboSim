
from colliders import BoxCollider, SphereCollider, CylinderCollider
import taichi as ti
from quaternion import quaternion
from collision import CollisionResponse
import taichi.math as tm


EPSILON = 1e-6

@ti.func
def box_v_box(
    box1: BoxCollider,
    box2: BoxCollider,
    position1: ti.types.vector(3, float),
    position2: ti.types.vector(3, float),
    orientation1: ti.types.vector(4, float),
    orientation2: ti.types.vector(4, float)
) -> CollisionResponse:
    
    # Get the face normals of each box
    axes       = getAxes(orientation1, orientation2)
    vertices_1 = getVertices(box1.half_extents, position1, orientation1 )
    vertices_2 = getVertices(box2.half_extents, position2, orientation2 )
    
    # Test each axis
    minOverlap = float('inf')
    direction  = 0.0
    for i in range(axes.n):
        axis = axes[i, :]
        if i >= 6:
            axis = axis.normalized()
            # Skip the axis if it is too small 
            # TODO: Is this necessary?
            if tm.length(axis) < EPSILON:
                continue
        
        # Project the boxes onto the axis
        projection_1_min, projection_1_max = project(vertices_1, axis)
        projection_2_min, projection_2_max = project(vertices_2, axis)
        
        # Check if the projections overlap
        overlap, dir = getOverlap(projection_1_min, 
                             projection_1_max, 
                             projection_2_min, 
                             projection_2_max)
        if overlap <= 0:
        
            return CollisionResponse(False)
        
        elif overlap < minOverlap:
            # Update the minimum overlap and collision normal
            minOverlap      = overlap
            direction       = dir
            normal = axis
    

    # Compute the penetration depth and contact points
    penetration = minOverlap
    r_1 = quaternion.rotate_vector(orientation1,  direction * normal) * minOverlap
    r_2 = quaternion.rotate_vector(orientation2, -direction * normal) * minOverlap
    
    return  CollisionResponse(
        True,
        normal,
        penetration,
        r_1,
        r_2
    )

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
    vertices = getVertices(box.half_extents, box_position, box_orientation)
    for i in range(axes.n):
        axis = axes[i]
        
        # Project the sphere onto the axis
        # TODO : Is this correct?
        projection_sphere_min = tm.dot(sphere_position - axis * sphere.radius, axis)
        projection_sphere_max = tm.dot(sphere_position + axis * sphere.radius, axis)
        
        # Project the box onto the axis
        projection_box_min, projection_box_max = project(vertices, axis)
        
        # Check if the projections overlap
        overlap, dir = getOverlap(projection_sphere_min,
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


@ti.func
def box_v_cylinder(
    box: BoxCollider,
    cylinder: CylinderCollider,
    box_position: ti.types.vector(3, float),
    cylinder_position: ti.types.vector(3, float),
    box_orientation: ti.types.vector(4, float),
    cylinder_orientation: ti.types.vector(4, float)
) -> CollisionResponse:
    
    # Get the box's face normals
    box_face_normals  = quaternion.to_rotation_matrix(box_orientation)
    
    # Add the cylinder's axis as an additional axis
    cylinder_axis = quaternion.rotate_vector(cylinder_orientation, ti.Vector([0.0, 0.0, 1.0]))
    
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 7 )
    axes[0:3, :] = box_face_normals
    axes[3,   :] = cylinder_axis
    
    
    
    # Add the cross products of the box's face normals and the cylinder's axis as additional axes
    for i in range(3):
        axes[3 + i,   :] = tm.cross(box_face_normals[i], cylinder_axis)
    
    # Test each axis
    minOverlap = float('inf')
    direction = 0.0
    vertices = getVertices(box.half_extents, box_position, box_orientation)

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
        
        projection_box_min, projection_box_max = project(vertices, axis)
        
        # Project the cylinder onto the axis
        projection_cylinder_min, projection_cylinder_max = project_cylinder(cylinder_axis, 
                                                                            top_center,
                                                                            bottom_center,
                                                                            cylinder.radius,
                                                                            axis)
        
        # Check if the projections overlap
        overlap, dir = getOverlap(projection_box_min,
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

@ti.func
def project_cylinder(
    cylinder_axis         : ti.types.vector(3, float),
    top_center            : ti.types.vector(3, float),
    bottom_center         : ti.types.vector(3, float),
    cylinder_radius       : float,
    axis                  : ti.types.vector(3, float)
) :
    
    # Calculate the center of the cylinder's top and bottom faces
    
    # Project the top and bottom centers onto the axis
    projection_top    = tm.dot(top_center, axis) 
    projection_bottom = tm.dot(bottom_center, axis)
    
    # Calculate the radius of the cylinder's projection onto the axis

    projection_radius = abs(cylinder_radius * tm.length(tm.cross(cylinder_axis, axis)))
    
    # Calculate the minimum and maximum values of the cylinder's projection onto the axis
    projection_min = min(projection_top, projection_bottom) - projection_radius
    projection_max = max(projection_top, projection_bottom) + projection_radius
    
    return (projection_min, projection_max)




@ti.func
def getAxes(orientation1: ti.types.vector(4, float), 
            orientation2: ti.types.vector(4, float)):
    # Get the face normals of each box
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 15 )

    rot_mat_1 = quaternion.to_rotation_matrix(orientation1)
    rot_mat_2 = quaternion.to_rotation_matrix(orientation2)

    axes[  : 3  , : ] = rot_mat_1
    axes[3 : 6  , : ] = rot_mat_2

    axes[6 : 15 , : ] = getEdgesAxes(rot_mat_1,rot_mat_2)    
    
    return axes



@ti.func
def project(
            vertices : ti.types.matrix(8,3,float),
            axis     : ti.types.vector(3,float)
            ):
   
   # Project each vertex of the box onto the axis
   minProjection =  float('inf')
   maxProjection = -float('inf')
   
   for i in range(vertices.n):
       vertex = vertices[i, :]
       projection    = tm.dot(vertex ,axis)
       minProjection = ti.min(minProjection, projection )
       maxProjection = ti.max(maxProjection, projection )
   
   return minProjection ,maxProjection

@ti.func
def getOverlap(projection_1_min, 
               projection_1_max, 
               projection_2_min, 
               projection_2_max):
    
    overlap = ti.min(projection_1_max, projection_2_max) - ti.max(projection_1_min, projection_2_min)

    # Direction of the overlap relative to the 1st box
    direction = 1.0 if projection_1_min < projection_2_min else -1.0

    return  overlap, direction
           

@ti.func
def getVertices(half_extents : ti.types.vector(3,float), 
                position     : ti.types.vector(3,float), 
                orientation  : ti.types.vector(4,float)):
   
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
def getEdgesAxes(
                edges_1 : ti.types.matrix(3, 3, float),
                edges_2 : ti.types.matrix(3, 3, float)
                 ) -> ti.types.matrix(9, 3, float):

    # Get the edge cross products of each box
    axes = ti.Matrix([ [0.0, 0.0, 0.0] ] * 9)

    for i in range(3): # 3 edges for box 1
        for j in range(3):
            axes[i+j, :] = tm.cross(edges_1[i, :], edges_2[j, :])

    return axes

    



   
