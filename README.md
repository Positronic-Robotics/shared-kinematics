# shared-kinematics

Arm models (kinematics and collision shapes) and the box and self-collision checks that the Positronic arm drivers
share.

The core is a header-only C++17 library on Eigen (`include/shared_kinematics`). A pybind11 module,
`shared_kinematics`, gives the same core to Python drivers. The core does not link a vendor library.

## What it does

A driver holds one `Guard` per arm. The guard knows the arm model, the box of the rig, and its limits.

- `clip(q_from, q_to, others)` runs once per new target, at policy rate, after the driver's IK. It walks the straight
  joint move from `q_from` to `q_to` and stops it at the first point where a part of the arm meets its floor: a
  side of the box, the self-collision clearance, or the clearance to another body. When a side or another body stops
  the move, one slide removes from the rest of the move its part along the binding checks, and walks again. The
  slid target keeps the joint limits of the URDF. The result is the committed target.
- `speed_scale(q, dq_step, dt, others)` runs once per tick of the fast loop. It gives one factor in [0, 1] on the
  whole joint step, so the step keeps its direction and stays on the straight joint line that clip checked.
- `distances(q, others)` gives each plane distance, each self-pair clearance and each clearance to another body, with
  its gradient in the joints. Use it for a plan check, for a summary, and for the IK push terms.

### The floors

- A side is a plane `(n, d)` in the rig frame: a point `x` is inside when `n . x + d >= 0`. Each arm reads the
  planes through its base pose `T_rig_base`.
- A part has a role. A `pad` may go past a side by the pad allowance of that side. A `rigid` part keeps a margin
  from every side. `Box` sets the allowances by the desk rule: a pad may go below a `surface` desk by the finger-bend
  travel of the gripper; every other allowance of a pad is 0; every rigid part keeps `margin`.
- The floor of each check is its limit, or its value at `q_from` when that is less. So a part that is already past
  its floor can move back in, and cannot move further out.
- Between two checked points of the walk, a check falls at most `path_tolerance` (1 mm) below its floor. The sample
  count comes from the per-joint rate bound of each check, with 16 samples at least, then 8 bisections.

### speed_scale

Near a floor, each part moves toward it at most `approach_gain` times its distance to the floor (`k`, 2.0/s). A pad
that may go below the desk meets the desk top at most at `touch_speed` (`v_touch`, 0.05 m/s); the limit falls
linearly to 0 at the pad floor. Above the desk top the limit is `v_touch + k * height`. Near another body the limit
is `k` times the distance to the clearance.

### Not configured

`box=None`, or a box with an empty side list, is the state "not configured". With it, and an empty `others` list:

- `speed_scale` returns 1 and names no side;
- `distances` reports no plane;
- `clip` checks the self pairs. On the FR3 its result is bit-equal to the walk of positronic-franka 0.12.3
  (`stop_before_self_collision`).

A driver maps an absent or disabled rig-config entry to `box=None` at start-up. A code change is not necessary.

## Python API

```python
import shared_kinematics as sk

model = sk.load_arm(urdf, sk.arm_table('fr3'), sk.gripper_table('robotiq_2f85'), attachments=(), base_pose=None)
box = sk.Box(
    sides=[
        sk.Side(name='desk', kind='surface', n=(0, 0, 1), d=0.0),
        sk.Side(name='front', kind='surface', n=(-1, 0, 0), d=0.8),
        sk.Side(name='back', kind='virtual', n=(1, 0, 0), d=0.3),
    ],
    finger_bend_travel=0.0,
    margin=0.005,
)
guard = sk.guard(model, box)  # box=None: not configured

result = guard.clip(q_from, q_to, others=[])  # -> ClipResult
scale = guard.speed_scale(q, dq_step, dt, others=[])  # -> ScaleResult
rows = guard.distances(q, others=[])  # -> Distances
deepest = sk.deepest_per_side(rows)  # -> dict[side, PlaneDistance]

# Two arms: each driver publishes its committed move; the other driver clips against it.
committed = guard.swept_capsules(q_now, result.q)  # list[WorldCapsule], rig frame
now = guard.capsules_at(q)  # list[WorldCapsule], rig frame, for speed_scale
```

| Name | Signature |
|---|---|
| `load_arm` | `(urdf: str, arm: ArmTable, gripper: GripperTable \| None = None, attachments: Sequence[Attachment] = (), base_pose: ndarray[4, 4] \| None = None) -> ArmModel` |
| `arm_table`, `gripper_table` | `(name: str) -> ArmTable`, `(name: str) -> GripperTable`: the bundled tables, for example `fr3` and `robotiq_2f85` |
| `guard` | `(model: ArmModel, box: Box \| None = None, params: GuardParams \| None = None) -> Guard` |
| `Guard.clip` | `(q_from, q_to, others: list[list[WorldCapsule]] = []) -> ClipResult` |
| `Guard.speed_scale` | `(q, dq_step, dt: float, others: list[list[WorldCapsule]] = []) -> ScaleResult` |
| `Guard.distances` | `(q, others = []) -> Distances` |
| `Guard.swept_capsules` | `(q_from, q_to) -> list[WorldCapsule]` |
| `Guard.capsules_at` | `(q) -> list[WorldCapsule]` |
| `ClipResult` | `q`, `fraction`, `bound_sides: list[str]`, `self_bound: bool`, `others_bound: list[int]`, `slid: bool`, `bound: bool` |
| `ScaleResult` | `s: float`, `slowest: str \| None` (a side name, or `other:<index>`) |
| `Distances` | `planes: list[PlaneDistance]`, `pairs: list[PairDistance]`, `others: list[OtherDistance]`; each row has its gradient |
| `GuardParams` | `path_tolerance=0.001`, `least_samples=16`, `bisections=8`, `approach_gain=2.0`, `touch_speed=0.05`, `other_clearance=0.05`, `swept_spacing=0.01`, `max_swept_samples=64` |
| `Box` | `sides: list[Side]`, `finger_bend_travel: float = 0.0`, `margin: float = 0.005` |
| `Side` | `name: str`, `kind: 'surface' \| 'virtual'`, `n: (x, y, z)`, `d: float` |
| `Attachment` | `name`, `mount_type`, `link`, `a`, `b`, `radius`, `pairs_with: list[str] \| None = None` |

Joint vectors are numpy arrays in the joint order of the arm table. Lengths are in metres, angles in radians.

## C++ API

`#include "shared_kinematics/shared_kinematics.hpp"`. The CMake target is `shared_kinematics::shared_kinematics`. The
wheel holds the headers too, in `shared_kinematics/include`.

```cpp
shared_kinematics::Chain chain(std::vector<Link> links);
shared_kinematics::ArmModel model{chain, bodies, self_collision, T_rig_base};
shared_kinematics::Guard guard(model, planes, GuardParams{});
ClipResult clip(const VectorXd& q_from, const VectorXd& q_to, const std::vector<OtherBody>& others = {}) const;
ScaleResult speed_scale(const VectorXd& q, const VectorXd& dq_step, double dt,
                        const std::vector<OtherBody>& others = {}) const;
Distances distances(const VectorXd& q, const std::vector<OtherBody>& others = {}) const;
OtherBody swept_capsules(const VectorXd& q_from, const VectorXd& q_to) const;
OtherBody capsules_at(const VectorXd& q) const;
```

The C++ side has no URDF parser. Python reads the URDF and builds the chain (`chain_from_urdf`).

## The arm model

- **Chain**: a tree of links with revolute joints, read from the URDF that the driver reports. A moving joint that
  the arm table does not name (for example a gripper finger) drops its links.
- **Arm table** (`python/shared_kinematics/arms/<arm>.json`): the joint names, the flange link, the capsules that hold
  the link meshes, and the self-collision model. `tools/fit_capsules.py` fits the capsules once, offline, from the
  collision meshes in `data/meshes`; a test checks that the capsules of each link cover its mesh. The self-collision
  model has its clearance, and optionally its own capsules and link pairs. With no capsules, it uses the arm
  capsules; with no pairs, it checks every link pair two or more joints apart.
- **FR3**: the self-collision model is the capsule set and the SRDF link pairs of franka_description, at 0.09 m, as
  positronic-franka 0.12.3 checks them, with its test for parallel axes (`parallel_test: franka32`). Every other
  self pair, an attachment pair too, tests parallel axes by their angle. The fitted capsules hold the link meshes for the box and the other bodies.
  `link0` is fixed to the mount and stays out of the plane checks. `link1` turns with joint 1, so the box bounds it.
- **Gripper table** (`grippers/<gripper>.json`): capsules in the flange axes that hold the gripper over its whole grip
  range. `robotiq_2f85`: two rigid body capsules and one pad capsule per finger.
- **Attachments**: rigid parts such as wrist cameras, one entry per mount type. Each joins the plane checks, and the
  self pairs with each link two or more joints away (or the links in `pairs_with`).

## Build and test

```bash
uv venv && uv pip install . pytest pytest-xdist trimesh scipy
.venv/bin/python -m pytest -n 4 python
```

The build finds Eigen 3.3 or later as a CMake package, or at `EIGEN3_INCLUDE_DIR`; else it fetches Eigen 3.4.0.

C++ tests and the timing run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I include -I tests -isystem /usr/include/eigen3 tests/test_guard.cpp -o test_guard
./test_guard
g++ -std=c++17 -O2 -DNDEBUG -I include -I tests -isystem /usr/include/eigen3 tests/bench_guard.cpp -o bench_guard
./bench_guard
```

### Parity

`tests/reference/positronic_franka_32.hpp` is a copy of the FR3 walk of positronic-franka 0.12.3. The C++ tests
compare `clip` to it on synthetic moves, with an empty box. `SK_PARITY_TRIPLES` names an optional file of recorded
walks: one line per walk, a label, then `from[7] to[7] result[7]`. The C++ and the Python tests then compare `clip`
to each recorded result.
