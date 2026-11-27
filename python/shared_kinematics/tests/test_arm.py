import numpy as np
import pytest
import shared_kinematics as sk
from conftest import FR3_URDF


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
    fr3 = sk.arm_table('fr3')
    table = fr3.model_copy(
        update={'self_collision': fr3.self_collision.model_copy(update={'capsules': None, 'pairs': None})}
    )
    model = sk.load_arm(FR3_URDF, table)
    pairs = {p.name for p in model.self_collision.pairs}
    assert 'link0-link2' in pairs and 'link5-link7' in pairs
    assert 'link0-link1' not in pairs and 'link6-link7' not in pairs
    assert len(model.self_collision.capsules) == len(fr3.capsules)


def test_a_wrist_camera_joins_the_box_and_the_self_pairs_with_links_two_or_more_joints_away():
    camera = sk.Attachment(
        name='wrist_camera',
        mount_type='zed_mini_bracket',
        link='link8',
        a=(0.06, 0, 0.02),
        b=(0.06, 0, 0.06),
        radius=0.02,
    )
    model = sk.load_arm(FR3_URDF, 'fr3', 'robotiq_2f85', [camera])
    assert model.bodies[-1].capsule.name == 'wrist_camera' and model.bodies[-1].role == sk.Role.RIGID
    camera_pairs = [p.name for p in model.self_collision.pairs if p.name.endswith('-wrist_camera')]
    assert camera_pairs == [f'link{k}-wrist_camera' for k in range(6)]
    picked = camera.model_copy(update={'pairs_with': ['link2']})
    model = sk.load_arm(FR3_URDF, 'fr3', None, [picked])
    assert [p.name for p in model.self_collision.pairs if 'camera' in p.name] == ['link2-wrist_camera']


def test_an_unknown_link_or_a_bad_base_pose_is_refused():
    with pytest.raises(ValueError, match='no link'):
        sk.load_arm(
            FR3_URDF,
            'fr3',
            None,
            [sk.Attachment(name='c', mount_type='m', link='nowhere', a=(0, 0, 0), b=(0, 0, 0), radius=0.01)],
        )
    with pytest.raises(ValueError):
        sk.load_arm(FR3_URDF, 'fr3', base_pose=np.full((4, 4), np.nan))
