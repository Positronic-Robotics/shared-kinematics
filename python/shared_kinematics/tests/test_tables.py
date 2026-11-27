import importlib.util

import numpy as np
import pytest
import shared_kinematics as sk
from conftest import REPO

trimesh = pytest.importorskip('trimesh')
pytest.importorskip('scipy')

MESHES = REPO / 'data' / 'meshes'


def fit_tool():
    spec = importlib.util.spec_from_file_location('fit_capsules', REPO / 'tools' / 'fit_capsules.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outside(points: np.ndarray, capsules) -> np.ndarray:
    """How far each point is outside the union of `capsules`, m; 0 or less is inside."""
    gap = np.full(len(points), np.inf)
    for c in capsules:
        a, b = np.array(c.a), np.array(c.b)
        ab = b - a
        t = np.clip((points - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
        gap = np.minimum(gap, np.linalg.norm(points - (a + t[:, None] * ab), axis=1) - c.radius)
    return gap


def surface(mesh, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=seed)
    return np.vstack([mesh.vertices, points])


@pytest.mark.parametrize('link', [f'link{k}' for k in range(8)])
def test_the_fr3_capsules_of_each_link_cover_its_collision_mesh(link):
    capsules = [c for c in sk.arm_table('fr3').capsules if c.link == link]
    assert capsules
    assert outside(surface(trimesh.load(MESHES / 'fr3' / f'{link}.stl'), seed=7), capsules).max() <= 0.0


@pytest.mark.parametrize('grip', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_robotiq_2f85_capsules_cover_its_meshes_over_the_grip_range(grip):
    table = sk.gripper_table('robotiq_2f85')
    tool = fit_tool()
    parts = tool.robotiq_2f85_parts(grip)
    body = [c for c in table.capsules if c.role == 'rigid']
    for side in ('left', 'right'):
        finger = [c for c in table.capsules if c.name == f'{side}_finger']
        assert [c.role for c in finger] == ['pad']
        for part in ('follower', 'pad', 'silicone_pad'):
            assert outside(parts[f'{side}_{part}'], finger).max() <= 0.0, f'{side}_{part}'
        for part in ('driver', 'coupler', 'spring_link'):
            assert outside(parts[f'{side}_{part}'], body).max() <= 0.0, f'{side}_{part}'
    for part in ('mount', 'base'):
        assert outside(parts[part], body).max() <= 0.0, part


def test_the_fr3_self_collision_table_is_the_srdf_pairs_at_0_09_m():
    own = sk.arm_table('fr3').self_collision
    assert own.clearance == 0.09
    assert own.pairs == [
        ('link0', 'link5'),
        ('link0', 'link6'),
        ('link0', 'link7'),
        ('link1', 'link5'),
        ('link1', 'link6'),
        ('link1', 'link7'),
        ('link2', 'link5'),
        ('link2', 'link7'),
    ]
    assert len(own.capsules) == 10


def test_the_base_links_of_the_fr3_stay_out_of_the_box_checks():
    table = sk.arm_table('fr3')
    assert {c.link for c in table.capsules if not c.in_planes} == {'link0', 'link1'}
