```python
@ti.dataclass
class Collision:
	body_1_idx : ti.types.i32
	body_2_idx : ti.types.i32
	
	virtual_material : Material
	
	collision : bool
	
	penetratrion_depth : ti.types.f32
	
	contact_normal : ti.types.vector(3, ti.types.f32) 
	
	active_points  : ti.types.u8 
	
	r_1 : ti.types.Matrix(4, 3, ti.types.f32)
	r_2 : ti.types.Matrix(4, 3, ti.types.f32)
	
	fricction_force : ti.types.vector(3, ti.types.f32)
	normal_force    : ti.types.vector(3, ti.types.f32)
	
	lambda_normal   : ti.types.f32
	lambda_friction : ti.types.f32
```
