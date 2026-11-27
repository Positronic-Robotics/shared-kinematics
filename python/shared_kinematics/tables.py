"""The capsule tables of the arms and the grippers, and the attachment entries of a rig."""

import json
from importlib import resources
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]
Length = Annotated[float, Field(ge=0.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class MountedCapsule(_Frozen):
    """A capsule in the axes of the link it is fixed to: the segment a-b grown by `radius`, in m."""

    name: str
    a: Vec3
    b: Vec3
    radius: Length
    role: Literal['pad', 'rigid'] = 'rigid'
    in_planes: bool = True  # False for a part that sits on the arm mount and cannot leave its place


class LinkCapsule(MountedCapsule):
    link: str


class SelfCapsule(_Frozen):
    link: str
    a: Vec3
    b: Vec3
    radius: Length


class SelfCollisionTable(_Frozen):
    """Capsules grouped by link, and the link pairs that keep `clearance` apart.

    No capsules: the arm capsules. No pairs: every pair of links two or more joints apart.
    """

    source: str
    clearance: float
    capsules: list[SelfCapsule] | None = None
    pairs: list[tuple[str, str]] | None = None


class ArmTable(_Frozen):
    """The capsules of one arm: `capsules` hold the link meshes (tools/fit_capsules.py fits them)."""

    arm: str
    joints: list[str]
    flange: str
    capsules: list[LinkCapsule]
    self_collision: SelfCollisionTable


class GripperTable(_Frozen):
    """The capsules of one gripper, in the flange axes of the arm, over its whole grip range."""

    gripper: str
    mount: str
    capsules: list[MountedCapsule]


class Attachment(_Frozen):
    """A rigid part fixed to a link that the arm model does not hold, for example a wrist camera."""

    name: str
    mount_type: str
    link: str
    a: Vec3
    b: Vec3
    radius: Length
    # The self-collision groups (link names) that the part keeps clear of; None: each link two or more joints away.
    pairs_with: list[str] | None = None


def arm_table(arm: str) -> ArmTable:
    """The table of a bundled arm, for example `fr3`."""
    text = resources.files(__package__).joinpath('arms', f'{arm}.json').read_text()
    return ArmTable.model_validate(json.loads(text))


def gripper_table(gripper: str) -> GripperTable:
    """The table of a bundled gripper, for example `robotiq_2f85`."""
    text = resources.files(__package__).joinpath('grippers', f'{gripper}.json').read_text()
    return GripperTable.model_validate(json.loads(text))
