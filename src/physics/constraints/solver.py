"""
    Description: File contains the general solver for the constraints of the physics engine. 
                (both for velocity and position based constraints )

    Author: Eduardo Lopez
    Organization: GIA-USB
"""
import taichi as ti


@ti.func
def solve_constraints(  bodies         : ti.template(), 
                        constraints    : ti.template(), 
                        h              : ti.types.f32, 
                        iteration      : ti.types.i32, 
                        position_based : ti.types.i32):
    """
        Solve the constraints of the physics engine. 

        Arguments:
        ----------
        bodies : ti.template()
            -> The bodies of the physics engine
        constraints : ti.template()
            -> The constraints of the physics engine
        h : ti.types.f32
            -> The timestep of the physics engine
        iteration : ti.types.i32
            -> The iteration of the solver
        position_based : ti.types.i32
            -> Whether the solver is position based or velocity based
    """

    for i in range(iteration):
        for constraint in constraints:
            if constraint.active:
                constraint.solve(bodies, h, position_based)