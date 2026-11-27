import numpy as np
import pytest
import shared_kinematics as sk
from conftest import FR3_URDF
from shared_kinematics.tables import SelfCollisionTable

FR3 = sk.arm_table('fr3')
GRIPPER = sk.gripper_table('robotiq_2f85')


def test_the_fr3_model_holds_the_arm_and_gripper_capsules_and_the_32_self_pairs(fr3):
    names = [b.capsule.name for b in fr3.bodies]
    assert names[:8] == [f'link{k}' for k in range(8)]
    assert {'left_finger', 'right_finger'} <= set(names)
    assert {b.capsule.name for b in fr3.bodies if b.role == sk.Role.PAD} == {'left_finger', 'right_finger'}
    assert [p.name for p in fr3.self_collision.pairs] == [
        'link0-link5',
        'link0-link6',
        'link0-link7',
        'link1-link5',
        'link1-link6',
        'link1-link7',
        'link2-link5',
        'link2-link7',
    ]
    assert fr3.self_collision.clearance == 0.09


def test_an_arm_table_with_no_pairs_checks_each_link_pair_two_or_more_joints_apart():
    own = FR3.self_collision
    table = sk.ArmTable(
        arm=FR3.arm,
        joints=FR3.joints,
        flange=FR3.flange,
        capsules=FR3.capsules,
        self_collision=SelfCollisionTable(source=own.source, clearance=own.clearance),
    )
    model = sk.load_arm(FR3_URDF, table)
    pairs = {p.name for p in model.self_collision.pairs}
    assert 'link0-link2' in pairs and 'link5-link7' in pairs
    assert 'link0-link1' not in pairs and 'link6-link7' not in pairs
    assert len(model.self_collision.capsules) == len(FR3.capsules)


def wrist_camera(pairs_with: list[str] | None = None) -> sk.Attachment:
    return sk.Attachment(
        name='wrist_camera',
        mount_type='zed_mini_bracket',
        link='link8',
        a=(0.06, 0, 0.02),
        b=(0.06, 0, 0.06),
        radius=0.02,
        pairs_with=pairs_with,
    )


def test_a_wrist_camera_joins_the_box_and_the_self_pairs_with_links_two_or_more_joints_away():
    model = sk.load_arm(FR3_URDF, FR3, GRIPPER, [wrist_camera()])
    assert model.bodies[-1].capsule.name == 'wrist_camera' and model.bodies[-1].role == sk.Role.RIGID
    camera_pairs = [p.name for p in model.self_collision.pairs if p.name.endswith('-wrist_camera')]
    assert camera_pairs == [f'link{k}-wrist_camera' for k in range(6)]
    model = sk.load_arm(FR3_URDF, FR3, None, [wrist_camera(pairs_with=['link2'])])
    assert [p.name for p in model.self_collision.pairs if 'camera' in p.name] == ['link2-wrist_camera']


def test_an_unknown_link_or_a_bad_base_pose_is_refused():
    with pytest.raises(ValueError, match='no link'):
        sk.load_arm(
            FR3_URDF,
            FR3,
            None,
            [sk.Attachment(name='c', mount_type='m', link='nowhere', a=(0, 0, 0), b=(0, 0, 0), radius=0.01)],
        )
    with pytest.raises(ValueError):
        sk.load_arm(FR3_URDF, FR3, base_pose=np.full((4, 4), np.nan))


def test_the_fr3_pairs_keep_the_parallel_test_of_32_and_an_attachment_pair_tests_by_angle():
    model = sk.load_arm(FR3_URDF, FR3, GRIPPER, [wrist_camera()])
    tests = {p.name: p.parallel for p in model.self_collision.pairs}
    assert tests['link0-link5'] == sk.ParallelTest.FRANKA_32
    assert tests['link0-wrist_camera'] == sk.ParallelTest.ANGLE
