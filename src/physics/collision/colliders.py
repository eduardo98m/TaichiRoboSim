import taichi as ti
from quaternion import quaternion

BOX         = 1
SPHERE      = 2
CYLINDER    = 3
PLANE       = 4
HEIGHTFIELD = 5



@ti.dataclass
class BoxCollider:
    """
    Class for box colliders
    
    Arguments:
    ----------
    `half_extents` : ti.types.vector(3, float)
        -> Half widths of the box
    """
    half_extents : ti.types.vector(3, float)

    @ti.func
    def compute_aabb(self, position, orientation):
        aabb = ti.Matrix.zero(ti.float32, 2, 3)
        aabb[0, :] = position - quaternion.rotate_vector(orientation, self.half_extents)
        aabb[1, :] = position + quaternion.rotate_vector(orientation, self.half_extents)
        return aabb

@ti.dataclass
class SphereCollider:
    """
    Class for sphere colliders
    
    Arguments:
    ----------
    `radius` : ti.types.f32
        -> Radius of the sphere
    """
    radius: ti.types.f32

    @ti.func
    def compute_aabb(self, position, orientation):
        aabb = ti.Matrix.zero(ti.float32, 2, 3)
        aabb[0, :] = position - self.radius
        aabb[1, :] = position + self.radius
        return aabb

@ti.dataclass
class CylinderCollider:
    """
    Class for cylinder colliders

    Arguments:
    ----------
    `radius` : ti.types.f32
        -> Radius of the cylinder
    
    `height` : ti.types.f32
        -> Height of the cylinder
    """

    radius: ti.types.f32
    height: ti.types.f32


    @ti.func
    def compute_aabb(self, position, orientation):
        aabb = ti.Matrix.zero(ti.float32, 2, 3)
        abb_vector = quaternion.rotate_vector(orientation, ti.Vector([self.radius, self.radius, self.height / 2])) 
        aabb[0, :] = position - abb_vector
        aabb[1, :] = position + abb_vector
        return aabb

@ti.dataclass
class PlaneCollider:
    """
    Class for plane colliders

    Arguments:
    ----------
    `normal` : ti.types.vector(3, float)
        -> Normal of the plane
    
    `offset` : ti.types.f32
        -> Offset of the plane
    """

    normal: ti.types.vector(3, float)
    offset: ti.types.f32




@ti.dataclass
class HeightFieldCollider:
    """
    Class for height field colliders

    Arguments:
    ----------
    `height_field` : ti.types.array
        -> Height field data
    """
    n_rows           : ti.types.u32 
    n_cols           : ti.types.u32
    height_field_id  : ti.types.u32
    x_coordinates_id : ti.types.u32
    y_coordinates_id : ti.types.u32
    aabb             : ti.types.matrix(2,3, float)
    type             : ti.types.u8 = HEIGHTFIELD

    def init(self):
        self.type  : ti.types.u8 = HEIGHTFIELD

    @ti.func
    def compute_aabb(self, position, orientation, height_field_data, x_coordinates, y_coordinates):
        self.aabb[0, :] = position - quaternion.rotate_vector(orientation, ti.Vector([x_coordinates[0],  y_coordinates[0], height_field_data[0,0]]))
        self.aabb[1, :] = position + quaternion.rotate_vector(orientation, ti.Vector([x_coordinates[-1], y_coordinates[-1], height_field_data[-1,-1]]))
    

@ti.dataclass
class Collider:
    """
    Class for dynamic colliders
    
    Arguments:
    ----------
    `type` : ti.types.u8
        -> Type of the collider
    """
    type : ti.types.u8
    box_collider : BoxCollider
    sphere_collider : SphereCollider
    cylinder_collider : CylinderCollider
    plane_collider : PlaneCollider
    aabb : ti.types.matrix(2,3, float)

    @ti.func
    def compute_aabb(self, position, orientation):
        if self.type == BOX:
            self.aabb = self.box_collider.compute_aabb(position, orientation)
        elif self.type == SPHERE:
            self.aabb = self.sphere_collider.compute_aabb(position, orientation)
        elif self.type == CYLINDER:
            self.aabb = self.cylinder_collider.compute_aabb(position, orientation)
        else:
            self.aabb = ti.Matrix.zero(ti.float32, 2, 3)







@ti.dataclass
class StaticCollider:
    """
    Class for static colliders
    
    Arguments:
    ----------
    `type` : ti.types.u8
        -> Type of the collider
    """
    type                  : ti.types.u8
    box_collider          : BoxCollider
    sphere_collider       : SphereCollider
    cylinder_collider     : CylinderCollider
    plane_collider        : PlaneCollider
    height_field_collider : HeightFieldCollider



if __name__ == "__main__":
    ti.init(ti.cpu)
    
    # Create a height field collider

    n_rows = 10
    n_cols = 10
    hf = HeightFieldCollider(
        n_rows = n_rows,
        n_cols = n_cols,
        height_field_ptr = 0,
        x_coordinates_ptr = 0,
        y_coordinates_ptr = 0,
        type = HEIGHTFIELD
    )
    




    


