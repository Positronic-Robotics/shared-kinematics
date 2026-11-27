"""Fit the bounded capsules of the arm and gripper tables to their collision meshes.

Run once when a mesh changes, then review and commit the tables:

    uv run --group dev python tools/fit_capsules.py

Each link or part gets one capsule, or two when two hold it in much less volume. A fitted capsule holds every mesh
vertex and a dense sample of the mesh surface; the tests check it again.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MESHES = ROOT / 'data' / 'meshes'
TABLES = ROOT / 'python' / 'shared_kinematics'
SAMPLES = 4000  # surface points per mesh, beside its vertices
SEED = 0
GROW = 0.0005  # m of radius over the fit, so that rounding cannot uncover a point
TWO_CAPSULES_IF_BELOW = 0.75  # the part of one capsule's volume that two must stay under


@dataclass
class Fit:
    a: np.ndarray
    b: np.ndarray
    radius: float

    @property
    def volume(self) -> float:
        return np.pi * self.radius**2 * np.linalg.norm(self.b - self.a) + 4 / 3 * np.pi * self.radius**3


def distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def capsule_on_line(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> Fit:
    """The least-volume capsule whose segment lies on the line through `origin` along unit `axis`."""
    rel = points - origin
    t = rel @ axis
    rho = np.linalg.norm(rel - t[:, None] * axis, axis=1)

    def segment(radius: float) -> tuple[float, float]:
        h = np.sqrt(np.maximum(radius**2 - rho**2, 0.0))
        lo, hi = np.max(t - h), np.min(t + h)  # the segment must reach t - h and t + h of each point
        return (hi, lo) if hi <= lo else ((hi + lo) / 2, (hi + lo) / 2)

    def volume(radius: float) -> float:
        lo, hi = segment(radius)
        return np.pi * radius**2 * (hi - lo) + 4 / 3 * np.pi * radius**3

    least = rho.max()
    most = max(least * 4, np.ptp(t)) + 1e-6
    radius = minimize_scalar(volume, bounds=(least, most), method='bounded', options={'xatol': 1e-6}).x
    lo, hi = segment(radius)
    return Fit(origin + lo * axis, origin + hi * axis, radius)


def fit_one(points: np.ndarray) -> Fit:
    # A capsule is convex: it holds the points when it holds the corners of their hull.
    points = points[ConvexHull(points).vertices]
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt  # rows: main axis, then two across it

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        axis = basis[0] + x[0] * basis[1] + x[1] * basis[2]
        return center + x[2] * basis[1] + x[3] * basis[2], axis / np.linalg.norm(axis)

    best = minimize(
        lambda x: capsule_on_line(points, *unpack(x)).volume,
        np.zeros(4),
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 4000},
    )
    return capsule_on_line(points, *unpack(best.x))


def covering(fit: Fit, points: np.ndarray) -> Fit:
    """`fit` rounded to 0.1 mm, its radius grown to hold every point."""
    a, b = np.round(fit.a, 4), np.round(fit.b, 4)
    radius = np.ceil((distance_to_segment(points, a, b).max() + GROW) * 1e4) / 1e4
    return Fit(a, b, float(radius))


def fit(points: np.ndarray) -> list[Fit]:
    one = covering(fit_one(points), points)
    center = points.mean(axis=0)
    main = np.linalg.svd(points - center, full_matrices=False)[2][0]
    t = (points - center) @ main
    best = [one]
    for cut in np.quantile(t, np.linspace(0.2, 0.8, 13)):
        parts = [points[t <= cut], points[t > cut]]
        if min(len(p) for p in parts) < 4:
            continue
        two = [covering(fit_one(p), p) for p in parts]
        if sum(f.volume for f in two) < min(TWO_CAPSULES_IF_BELOW * one.volume, sum(f.volume for f in best)):
            best = two
    return best


def mesh_points(mesh: trimesh.Trimesh) -> np.ndarray:
    surface, _ = trimesh.sample.sample_surface(mesh, SAMPLES, seed=SEED)
    return np.vstack([mesh.vertices, surface])


def transform(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def turn(axis, angle) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(np.asarray(axis, float) * angle).as_matrix()
    return t


def moved(mesh: trimesh.Trimesh, pose: np.ndarray) -> np.ndarray:
    return mesh_points(mesh) @ pose[:3, :3].T + pose[:3, 3]


# The Robotiq 2F-85 on the FR3 flange, as positronic's model grafts it: (link, parent, xyz, rpy, grip axis, mesh,
# mesh offset). A joint with a grip axis turns by grip * 0.8 rad.
GRIP_TRAVEL = 0.8
GRIP_STEPS = 11
HALF_PI = 1.5707963268


def robotiq_2f85_rows(side: str, sign: int, base_yaw: float) -> list[tuple]:
    return [
        (f'{side}_driver', 'base', (0, sign * 0.0306011, 0.054904), (0, 0, base_yaw), (1, 0, 0), 'driver', None),
        (f'{side}_coupler', f'{side}_driver', (0, 0.0315, -0.0041), (0, 0, 0), None, 'coupler', None),
        (f'{side}_spring_link', 'base', (0, sign * 0.0132, 0.0609), (0, 0, base_yaw), (1, 0, 0), 'spring_link', None),
        (
            f'{side}_follower',
            f'{side}_spring_link',
            (0, 0.037, 0.044),
            (0, 0, 0),
            (-1, 0, 0),
            'follower',
            (0, 0.018, -0.0065),
        ),
        (f'{side}_pad', f'{side}_follower', (0, -0.0009, 0.00702), (0, 0, 0), None, 'pad', None),
        (f'{side}_silicone_pad', f'{side}_pad', (0, 0, 0), (0, 0, 0), None, 'silicone_pad', None),
    ]


def robotiq_2f85_parts(grip: float) -> dict[str, np.ndarray]:
    """The mesh points of each 2F-85 part in the flange frame at `grip` (0 open, 1 closed)."""
    meshes = {p.stem: trimesh.load(p) for p in (MESHES / 'robotiq_2f85').glob('*.stl')}
    mount = transform((0, 0, 0.007), (0, 0, HALF_PI))
    poses = {'mount': mount, 'base': mount @ transform((0, 0, 0.0038), (0, 0, -HALF_PI))}
    points = {'mount': moved(meshes['base_mount'], poses['mount']), 'base': moved(meshes['base'], poses['base'])}
    for side, sign, yaw in (('right', 1, 0.0), ('left', -1, np.pi)):
        for link, parent, xyz, rpy, axis, mesh, offset in robotiq_2f85_rows(side, sign, yaw):
            pose = poses[parent] @ transform(xyz, rpy)
            if axis is not None:
                pose = pose @ turn(axis, grip * GRIP_TRAVEL)
            poses[link] = pose
            points[link] = moved(meshes[mesh], pose @ transform(offset or (0, 0, 0), (0, 0, 0)))
    return points


def rows(name: str, link: str | None, fits: list[Fit], role: str, in_planes: bool) -> list[dict]:
    """Table rows for `fits`. A gripper row names no link: it is fixed to the flange of the arm."""
    return [
        {
            'name': name if len(fits) == 1 else f'{name}_{i}',
            **({'link': link} if link else {}),
            'a': [round(float(x), 4) for x in f.a],
            'b': [round(float(x), 4) for x in f.b],
            'radius': f.radius,
            'role': role,
            'in_planes': in_planes,
        }
        for i, f in enumerate(fits)
    ]


def fr3_capsules() -> list[dict]:
    out = []
    for k in range(8):
        points = mesh_points(trimesh.load(MESHES / 'fr3' / f'link{k}.stl'))
        # link0 and link1 sit on the mount: their place does not change, so the box does not bound them.
        out += rows(f'link{k}', f'link{k}', fit(points), 'rigid', in_planes=k >= 2)
    return out


def robotiq_2f85_capsules() -> list[dict]:
    grips = [robotiq_2f85_parts(g) for g in np.linspace(0.0, 1.0, GRIP_STEPS)]
    body = ['mount', 'base'] + [f'{s}_{p}' for s in ('left', 'right') for p in ('driver', 'coupler', 'spring_link')]
    out = rows('body', None, fit(np.vstack([g[p] for g in grips for p in body])), 'rigid', True)
    for side in ('left', 'right'):
        finger = [f'{side}_{p}' for p in ('follower', 'pad', 'silicone_pad')]
        points = np.vstack([g[p] for g in grips for p in finger])
        out += rows(f'{side}_finger', None, [covering(fit_one(points), points)], 'pad', True)
    return out


def update(path: Path, capsules: list[dict]) -> None:
    table = json.loads(path.read_text())
    table['capsules'] = capsules
    path.write_text(json.dumps(table, indent=2) + '\n')
    print(f'{path.relative_to(ROOT)}: {len(capsules)} capsules')
    for c in capsules:
        print(f'  {c["name"]:>14} {c.get("link", "flange"):>6} r {c["radius"] * 1e3:5.1f} mm  a {c["a"]}  b {c["b"]}')


def main() -> None:
    update(TABLES / 'arms' / 'fr3.json', fr3_capsules())
    update(TABLES / 'grippers' / 'robotiq_2f85.json', robotiq_2f85_capsules())
    for grip in (0.0, 1.0):
        parts = robotiq_2f85_parts(grip)
        tips = np.vstack([parts[f'{side}_silicone_pad'] for side in ('left', 'right')])
        print(
            f'grip {grip}: pad tips reach z {tips[:, 2].max() * 1e3:.1f} mm, y {tips[:, 1].min() * 1e3:.1f} .. '
            f'{tips[:, 1].max() * 1e3:.1f} mm'
        )


if __name__ == '__main__':
    main()
