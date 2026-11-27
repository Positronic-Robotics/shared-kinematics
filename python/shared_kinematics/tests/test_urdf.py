import numpy as np
import pytest
import shared_kinematics as sk
from conftest import FR3_URDF

FR3_JOINTS = [f'joint{i}' for i in range(1, 8)]
# Franka's modified DH parameters of joints 1 .. 7 and of the flange.
DH_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
DH_D = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
DH_ALPHA = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def dh_frames(q: np.ndarray) -> list[np.ndarray]:
    """link1 .. link7 and the flange from the modified DH parameters."""

    def rot(axis: int, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        t = np.eye(4)
        t[i, i], t[i, j], t[j, i], t[j, j] = c, -s, s, c
        return t

    def move(x: float, z: float) -> np.ndarray:
        t = np.eye(4)
        t[0, 3], t[2, 3] = x, z
        return t

    frames, t = [], np.eye(4)
    for i in range(8):
        t = t @ rot(0, DH_ALPHA[i]) @ move(DH_A[i], 0) @ rot(2, q[i] if i < 7 else 0.0) @ move(0, DH_D[i])
        frames.append(t)
    return frames


def test_the_fr3_chain_from_the_urdf_matches_the_dh_frames_within_0_5_mm():
    chain = sk.chain_from_urdf(FR3_URDF, FR3_JOINTS)
    links = [chain.find(f'link{k}') for k in range(1, 9)]
    rng = np.random.default_rng(0)
    worst_mm, worst_rad = 0.0, 0.0
    for _ in range(500):
        q = rng.uniform(LOWER, UPPER)
        ours = chain.frames(q)
        for k, theirs in zip(links, dh_frames(q), strict=True):
            worst_mm = max(worst_mm, np.linalg.norm(ours[k][:3, 3] - theirs[:3, 3]) * 1e3)
            turn = ours[k][:3, :3].T @ theirs[:3, :3]
            worst_rad = max(worst_rad, np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))
    assert worst_mm < 0.5
    assert worst_rad < 1e-6


def test_a_moving_joint_that_the_list_does_not_name_drops_its_links():
    finger = (
        '<link name="finger"/><joint name="finger_joint" type="revolute"><parent link="link8"/>'
        '<child link="finger"/><axis xyz="1 0 0"/></joint>'
    )
    chain = sk.chain_from_urdf(FR3_URDF.replace('</robot>', finger + '</robot>'), FR3_JOINTS)
    assert chain.joint_count == 7
    assert chain.find('finger') == -1
    assert chain.find('end_effector') > 0


def test_a_joint_that_is_not_revolute_or_missing_is_refused():
    with pytest.raises(ValueError, match='no joints'):
        sk.chain_from_urdf(FR3_URDF, [*FR3_JOINTS, 'joint9'])
    prismatic = FR3_URDF.replace('name="joint7" type="revolute"', 'name="joint7" type="prismatic"')
    assert prismatic != FR3_URDF
    with pytest.raises(ValueError, match='revolute joints only'):
        sk.chain_from_urdf(prismatic, FR3_JOINTS)


def test_a_joint_with_no_axis_turns_about_x():
    urdf = (
        '<robot name="r"><link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/></joint></robot>'
    )
    chain = sk.chain_from_urdf(urdf, ['j'])
    np.testing.assert_allclose(chain.links[1].axis, [1, 0, 0])
