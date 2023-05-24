

import taichi as ti
import taichi.math as tm
from quaternion import quaternion



@ti.kernel
def test():
    # Create a taichi matrix
    

    a = ti.Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])

    for i in range(a.n):
        vec = ti.Vector([a[i,0], a[i,1], a[i,2]])

        print(vec @ a[i, :])
    
    mat = ti.Matrix([ [0.0, 0.0, 0.0] ]  * 15)
    print(mat.n)
    for i in range(mat.n):
        mat[i, :] = ti.Vector([-1.0, i, i])
        mat[i, :] = tm.cross( mat[i, :], ti.Vector([0.0, 1.0, 0.0]))
        print(mat[i, :])

if __name__ == "__main__":
    ti.init(ti.cpu)
    test()
