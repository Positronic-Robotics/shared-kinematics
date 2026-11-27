"""The box of a rig: its sides in the rig frame, and the allowances of the desk rule."""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ._core import ArmModel, Distances, Guard, GuardParams, Plane, PlaneDistance, SideKind

DESK = 'desk'
_KINDS = {'surface': SideKind.SURFACE, 'virtual': SideKind.VIRTUAL}


class Side(BaseModel):
    """One side: a point x of the rig frame is inside when n . x + d >= 0."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    kind: Literal['surface', 'virtual']
    n: tuple[float, float, float]
    d: float


class Box(BaseModel):
    """The sides of a rig and the desk rule.

    A pad may go below a `surface` desk by `finger_bend_travel`; every other part keeps `margin` from every side.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    sides: list[Side]
    finger_bend_travel: Annotated[float, Field(ge=0.0)] = 0.0
    margin: Annotated[float, Field(ge=0.0)] = 0.005

    def planes(self) -> list[Plane]:
        return [
            Plane(
                side.name,
                _KINDS[side.kind],
                np.array(side.n),
                side.d,
                self.finger_bend_travel if side.name == DESK and side.kind == 'surface' else 0.0,
                -self.margin,
            )
            for side in self.sides
        ]


def guard(model: ArmModel, box: Box | None = None, params: GuardParams | None = None) -> Guard:
    """The guard of one arm. With no box, clip keeps the self pairs and speed_scale returns 1."""
    planes = [] if box is None else box.planes()
    return Guard(model, planes, GuardParams() if params is None else params)


def deepest_per_side(distances: Distances) -> dict[str, PlaneDistance]:
    """For each side, the part nearest to its floor or deepest past it."""
    deepest: dict[str, PlaneDistance] = {}
    for row in distances.planes:
        held = deepest.get(row.side)
        if held is None or row.distance - row.floor < held.distance - held.floor:
            deepest[row.side] = row
    return deepest
