"""Build an `ArmModel` from a URDF and the capsule tables."""

from collections.abc import Sequence

import numpy as np

from ._core import ArmModel, Body, Capsule, Chain, Role, SelfCollision, SelfPair
from .tables import ArmTable, Attachment, GripperTable, MountedCapsule
from .urdf import chain_from_urdf

_ROLES = {'pad': Role.PAD, 'rigid': Role.RIGID}


def _link(chain: Chain, name: str) -> int:
    index = chain.find(name)
    if index < 0:
        raise ValueError(f'the chain has no link {name}')
    return index


def _body(chain: Chain, link: str, capsule: MountedCapsule) -> Body:
    return Body(
        Capsule(capsule.name, _link(chain, link), np.array(capsule.a), np.array(capsule.b), capsule.radius),
        _ROLES[capsule.role],
        capsule.in_planes,
    )


def joints_between(chain: Chain, a: int, b: int) -> int:
    """The count of joints on the path between links a and b."""
    return len(set(chain.moving_joints(a)) ^ set(chain.moving_joints(b)))


def pairs_two_joints_apart(chain: Chain, links: Sequence[str]) -> list[tuple[str, str]]:
    """Every pair of `links` with two or more joints between them, in the order of `links`."""
    index = {name: _link(chain, name) for name in links}
    ordered = sorted(set(links), key=index.__getitem__)
    return [
        (a, b)
        for i, a in enumerate(ordered)
        for b in ordered[i + 1 :]
        if joints_between(chain, index[a], index[b]) >= 2
    ]


def _self_collision(chain: Chain, table: ArmTable, attachments: Sequence[Attachment]) -> SelfCollision:
    own = table.self_collision
    if own.capsules is not None:
        rows = [(c.link, c.link, c.a, c.b, c.radius) for c in own.capsules]
    else:
        rows = [(c.link, c.link, c.a, c.b, c.radius) for c in table.capsules]
    pairs = list(own.pairs) if own.pairs is not None else pairs_two_joints_apart(chain, [r[0] for r in rows])
    clearance = own.clearance
    groups = {link for link, *_ in rows}
    for attachment in attachments:
        at = _link(chain, attachment.link)
        others = attachment.pairs_with
        if others is None:
            reach = [g for g in groups if joints_between(chain, at, _link(chain, g)) >= 2]
            others = sorted(reach, key=lambda g: _link(chain, g))
        if attachment.name in groups:
            raise ValueError(f'attachment {attachment.name} has the name of a link group')
        rows.append((attachment.name, attachment.link, attachment.a, attachment.b, attachment.radius))
        pairs += [(g, attachment.name) for g in others]
    capsules = [Capsule(group, _link(chain, link), np.array(a), np.array(b), r) for group, link, a, b, r in rows]
    names = [group for group, *_ in rows]
    self_pairs = []
    for first, second in pairs:
        a = [i for i, n in enumerate(names) if n == first]
        b = [i for i, n in enumerate(names) if n == second]
        if not a or not b:
            raise ValueError(f'self pair {first}-{second} names a group with no capsule')
        self_pairs.append(SelfPair(f'{first}-{second}', a, b))
    return SelfCollision(capsules, self_pairs, clearance)


def load_arm(
    urdf: str,
    arm: ArmTable,
    gripper: GripperTable | None = None,
    attachments: Sequence[Attachment] = (),
    base_pose: np.ndarray | None = None,
) -> ArmModel:
    """The model of one arm.

    `urdf` is the arm's URDF text, as its driver reports it. `arm_table` and `gripper_table` load a bundled table.
    `base_pose` is T_rig_base, the arm base in the rig frame (identity when None). The gripper capsules join the box
    checks; the attachments join the box checks and the self pairs.
    """
    chain = chain_from_urdf(urdf, arm.joints)
    bodies = [_body(chain, c.link, c) for c in arm.capsules]
    if gripper is not None:
        bodies += [_body(chain, arm.flange, c) for c in gripper.capsules]
    bodies += [_body(chain, a.link, MountedCapsule(name=a.name, a=a.a, b=a.b, radius=a.radius)) for a in attachments]
    pose = np.eye(4) if base_pose is None else np.asarray(base_pose, dtype=float)
    return ArmModel(chain, bodies, _self_collision(chain, arm, attachments), pose)
