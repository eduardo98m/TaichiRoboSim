```python
@ti.dataclass
class body:
	position    : ti.types.vector(3, ti.types.f32)
	velocity    : ti.types.vector(3, ti.types.f32)
	orientation : ti.types.vector(4, ti.types.f32)
	angular_velocity : ti.types.vector(3, ti.types.f32)
	
	mass        : ti.types.f32
	inertia     : ti.types.vector(6, ti.types.f32)
	
	fixed       : bool
	
	dynamic_inertia     : ti.types.matrix(3,3, ti.types.f32)
	inv_dynamic_inertia : ti.types.matrix(3,3, ti.types.f32)
	
	collider    : Collider
	material    : Material
	
```
