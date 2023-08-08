
from bodies import RigidBody
from typing import Union
from .constraint import Constraint
import taichi as ti
from constraint_types import HINGE_JOINT
from physics.constraints.hinge_joint_constraint import compute_hinge_joint_constraint

@ti.dataclass
def handle_constraint(body_1 : RigidBody, 
                      body_2 : RigidBody, 
                      constraint: Constraint): 
    """
        Handle the constraint of the two bodies. 
    """

    if constraint.type == HINGE_JOINT:
        compute_hinge_joint_constraint(body_1, body_2, constraint)