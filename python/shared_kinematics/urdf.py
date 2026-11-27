"""Build a `Chain` from a URDF: the links that the named joints and fixed joints reach from the root."""

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np

from ._core import Chain, Link


def _floats(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.array([float(x) for x in text.split()] if text else default, dtype=float)


def origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """The 4x4 transform of a URDF origin: translation xyz, then fixed-axis roll, pitch, yaw."""
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    t = np.eye(4)
    t[:3, :3] = rz @ ry @ rx
    t[:3, 3] = xyz
    return t


def _link_of(joint: ET.Element, child: str, parent: int, index: int) -> Link:
    """The link `child` below `joint`, whose parent link is link `parent` of the chain and whose joint is `index`."""
    if index >= 0 and joint.attrib.get('type') not in ('revolute', 'continuous'):
        raise ValueError(
            f'joint {joint.attrib["name"]} is {joint.attrib.get("type")}; the chain takes revolute joints only'
        )
    origin, axis, limit = joint.find('origin'), joint.find('axis'), joint.find('limit')
    xyz = _floats(origin.get('xyz') if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _floats(origin.get('rpy') if origin is not None else None, (0.0, 0.0, 0.0))
    turn = _floats(axis.get('xyz') if axis is not None else None, (1.0, 0.0, 0.0))
    bounded = index >= 0 and joint.attrib.get('type') == 'revolute' and limit is not None
    lower = float(limit.get('lower', 0.0)) if bounded else -math.inf
    upper = float(limit.get('upper', 0.0)) if bounded else math.inf
    return Link(child, parent, origin_matrix(xyz, rpy), index, turn, lower, upper)


def chain_from_urdf(urdf: str, joints: Sequence[str]) -> Chain:
    """The chain of `urdf` whose joint i is the URDF joint `joints[i]`.

    A link below a moving joint that `joints` does not name is left out, for example a gripper finger.
    """
    robot = ET.fromstring(urdf)
    children: dict[str, list[tuple[ET.Element, str]]] = {}
    child_links = set()
    for joint in robot.findall('joint'):
        parent, child = joint.find('parent'), joint.find('child')
        if parent is None or child is None:
            raise ValueError(f'joint {joint.get("name")} has no parent or child')
        children.setdefault(parent.attrib['link'], []).append((joint, child.attrib['link']))
        child_links.add(child.attrib['link'])
    roots = [link.attrib['name'] for link in robot.findall('link') if link.attrib['name'] not in child_links]
    if len(roots) != 1:
        raise ValueError(f'the URDF has {len(roots)} root links, not one')
    index = {name: i for i, name in enumerate(joints)}
    links = [Link(roots[0], -1)]
    found: set[str] = set()
    queue = [(roots[0], 0)]
    while queue:
        parent_name, parent = queue.pop(0)
        for joint, child in children.get(parent_name, []):
            name = joint.attrib['name']
            if joint.attrib.get('type') != 'fixed' and name not in index:
                continue
            links.append(_link_of(joint, child, parent, index.get(name, -1)))
            found.add(name)
            queue.append((child, len(links) - 1))
    missing = [name for name in joints if name not in found]
    if missing:
        raise ValueError(f'the URDF has no joints {missing} under its root')
    return Chain(links)
