```python
@ti.dataclass
class HingeJoint:
	axes_1 : ti.Matrix(3, 3, ti.float32)
	axes_2 : ti.Matrix(3, 3, ti.float32)
	
	stiffness  : ti.float32
	damping    : ti.float32
	
	limit_angle : bool
	angle_limit : ti.float32
	
	constraint : AngularConstraint
```

