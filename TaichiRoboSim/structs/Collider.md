Collider struct, this struct has all the attributes of all the types of dynamic colliders, but can only represent one type of collider at a time.

```python
@ti.dataclass
class Collider:
	type: ti.u8
	sphere  : Sphere
	box     : Box
	capsule : Capsule
	aabb    : ti.types.matrix(2, 3, ti.types.f32)
```
