

import taichi as ti
import taichi.math as tm
from quaternion import quaternion

@ti.func
def mutability(a):
    a[0, 0] = 69.0

@ti.data_oriented
class TestClass:
    def __init__(self) -> None:
        self.a = ti.field(ti.f32, shape = (3, 3))
    
    @ti.func
    def mutate(self):
        self.a[0, 0] = self.a[0, 0] + 69.0

@ti.dataclass
class Wheel:
    radius : ti.float32
    width  : ti.float32


@ti.dataclass
class Car:
    power : ti.float32
    wheel : Wheel

        

@ti.kernel
def test():
    # Create a taichi matrix
    

    # Create a wheel
    wheel = Wheel(radius = 0.5, width = 0.2)
    # Create a car
    car = Car(power = 100.0, wheel = wheel)

    

if __name__ == "__main__":
    ti.init(ti.cpu)
    test_class = TestClass()
    print(ti.Vector([1,2,3], dt=ti.f32))
    test()
