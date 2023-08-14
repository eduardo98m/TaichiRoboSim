```python
@ti.dataclass
class PositionalConstraint:
	r_1           : ti.Vector(3, ti.float32)
	r_2           : ti.Vector(3, ti.float32)
	
	compliance    : ti.float32
	lagrange_mult : ti.float32
	
	body_1_idx : ti.int32
	body_2_idx : ti.int32
	
	force     : ti.Vector(3, ti.float32)
```
