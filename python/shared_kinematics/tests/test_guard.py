import os
from pathlib import Path

import numpy as np
import pytest
import shared_kinematics as sk
from conftest import FR3_URDF

LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def desk_and_walls(z: float = 0.05, **kwargs) -> sk.Box:
    return sk.Box(
        sides=[
            sk.Side(name='desk', kind='surface', n=(0, 0, 1), d=-z),
            sk.Side(name='front', kind='surface', n=(-1, 0, 0), d=0.9),
            sk.Side(name='back', kind='virtual', n=(1, 0, 0), d=0.35),
        ],
        **kwargs,
    )


def test_with_no_box_the_guard_changes_nothing_but_the_self_pairs(fr3):
    rng = np.random.default_rng(1)
    for box in (None, sk.Box(sides=[])):
        guard = sk.guard(fr3, box)
        assert guard.planes == []
        for _ in range(50):
            q = rng.uniform(LOWER, UPPER)
            dq = rng.normal(0, 0.001, 7)
            scale = guard.speed_scale(q, dq, 0.001)
            assert scale.s == 1.0 and scale.slowest is None
            assert guard.distances(q).planes == []
            result = guard.clip(q, q + 10 * dq)
            assert result.bound_sides == [] and result.others_bound == [] and not result.slid
            assert result.bound == result.self_bound


def test_the_desk_rule_lets_pads_below_a_surface_desk_and_keeps_rigid_parts_at_the_margin():
    planes = {p.name: p for p in desk_and_walls(finger_bend_travel=0.008, margin=0.004).planes()}
    assert planes['desk'].pad_allowance == 0.008
    assert planes['front'].pad_allowance == 0.0 and planes['back'].pad_allowance == 0.0
    assert {p.rigid_allowance for p in planes.values()} == {-0.004}
    assert planes['back'].kind == sk.SideKind.VIRTUAL
    virtual_desk = sk.Box(sides=[sk.Side(name='desk', kind='virtual', n=(0, 0, 1), d=0.0)], finger_bend_travel=0.01)
    assert virtual_desk.planes()[0].pad_allowance == 0.0


def test_a_box_refuses_a_negative_travel_or_margin():
    with pytest.raises(ValueError):
        sk.Box(sides=[], finger_bend_travel=-0.001)
    with pytest.raises(ValueError):
        sk.Box(sides=[], margin=-0.001)


def test_the_planes_stay_in_the_rig_frame_and_each_arm_reads_them_through_its_base_pose(fr3, low):
    raised = np.eye(4)
    raised[2, 3] = 0.1
    lifted = sk.load_arm(FR3_URDF, 'fr3', 'robotiq_2f85', base_pose=raised)
    box = desk_and_walls(z=0.0)
    at_origin = {r.body: r.distance for r in sk.guard(fr3, box).distances(low).planes if r.side == 'desk'}
    at_lift = {r.body: r.distance for r in sk.guard(lifted, box).distances(low).planes if r.side == 'desk'}
    for body, distance in at_origin.items():
        assert at_lift[body] == pytest.approx(distance + 0.1, abs=1e-12)


def test_the_deepest_part_per_side_is_the_one_nearest_its_floor(fr3, low):
    distances = sk.guard(fr3, desk_and_walls()).distances(low)
    deepest = sk.deepest_per_side(distances)
    assert set(deepest) == {'desk', 'front', 'back'}
    for side, row in deepest.items():
        assert row.distance - row.floor == min(r.distance - r.floor for r in distances.planes if r.side == side)


def test_a_target_through_the_desk_stops_at_the_desk_and_ends_bounded(fr3, low):
    guard = sk.guard(fr3, desk_and_walls(z=0.05, finger_bend_travel=0.005))
    target = low.copy()
    target[1] += 0.35  # the shoulder forward: the hand goes down and out
    result = guard.clip(low, target)
    assert result.bound and 'desk' in result.bound_sides
    for s in np.linspace(0, 1, 400):
        q = low + s * (result.q - low)
        for row in guard.distances(q).planes:
            assert row.distance >= row.floor - 0.001


def test_two_arms_keep_their_clearance_across_the_other_arms_committed_move(fr3, low):
    facing = np.eye(4)
    facing[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    facing[0, 3] = 1.4
    a, b = sk.guard(fr3), sk.guard(sk.load_arm(FR3_URDF, 'fr3', 'robotiq_2f85', base_pose=facing))
    reach = low.copy()
    reach[1] += 0.5
    reach[3] += 0.6
    b_move = b.swept_capsules(low, low + 0.3 * (reach - low))
    result = a.clip(low, reach, [b_move])
    assert result.others_bound == [0]
    for s in np.linspace(0, 1, 100):
        for k in np.linspace(0, 0.3, 16):
            other = b.capsules_at(low + k * (reach - low))
            least = min(r.clearance for r in a.distances(low + s * (result.q - low), [other]).others)
            assert least >= a.params.other_clearance - 0.001


TRIPLES = os.environ.get('SK_PARITY_TRIPLES')


@pytest.mark.skipif(not TRIPLES, reason='SK_PARITY_TRIPLES names no file of recorded walks')
def test_with_no_box_clip_equals_the_recorded_walks_of_32(fr3):
    guard = sk.guard(fr3)
    rows = np.loadtxt(Path(TRIPLES), usecols=range(1, 22))
    worst, equal = 0.0, 0
    for row in rows:
        q = guard.clip(row[:7], row[7:14]).q
        worst = max(worst, np.abs(q - row[14:]).max())
        equal += bool(np.array_equal(q, row[14:]))
    print(f'{len(rows)} recorded walks, {equal} bit-equal, worst {worst} rad')
    assert worst < 1e-9
