"""Time clip and speed_scale through the Python module on the FR3 with a Robotiq 2F-85.

uv run python tools/time_guard.py
"""

import time
from pathlib import Path

import numpy as np
import shared_kinematics as sk

ROOT = Path(__file__).resolve().parents[1]
LOW = np.array([0.0, 0.2, 0.0, -2.2, 0.0, 2.4, np.pi / 4])
PARK = np.array([0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78])
BOX = sk.Box(
    sides=[
        sk.Side(name='desk', kind='surface', n=(0, 0, 1), d=-0.05),
        sk.Side(name='front', kind='surface', n=(-1, 0, 0), d=0.95),
        sk.Side(name='left', kind='surface', n=(0, -1, 0), d=0.6),
        sk.Side(name='right', kind='surface', n=(0, 1, 0), d=0.6),
        sk.Side(name='back', kind='virtual', n=(1, 0, 0), d=0.35),
    ],
    finger_bend_travel=0.005,
)


def report(what: str, calls) -> None:
    us = []
    for call in calls:
        t0 = time.perf_counter()
        call()
        us.append((time.perf_counter() - t0) * 1e6)
    p50, p99 = np.percentile(us, [50, 99])
    print(f'{what:<52} median {p50:8.1f} us, p99 {p99:8.1f} us, max {max(us):8.1f} us ({len(us)} calls)')


def time_guard(name: str, guard: sk.Guard, rng: np.random.Generator) -> None:
    starts = LOW + rng.normal(0, 0.3, (2000, 7))
    steps = rng.normal(0, 0.02, (2000, 7))
    ticks = LOW + rng.normal(0, 0.3, (10000, 7))
    tick_steps = rng.normal(0, 0.0005, (10000, 7))
    report(
        f'clip, policy-rate target, {name}',
        [lambda q=q, d=d: guard.clip(q, q + d) for q, d in zip(starts, steps, strict=True)],
    )
    report(f'clip, park move, {name}', [lambda q=q: guard.clip(q, PARK) for q in starts[:200]])
    report(
        f'speed_scale, one tick, {name}',
        [lambda q=q, d=d: guard.speed_scale(q, d, 0.001) for q, d in zip(ticks, tick_steps, strict=True)],
    )


def main() -> None:
    model = sk.load_arm((ROOT / 'data' / 'fr3.urdf').read_text(), 'fr3', 'robotiq_2f85')
    rng = np.random.default_rng(7)
    time_guard('no box', sk.guard(model), rng)
    time_guard('box of 5 sides', sk.guard(model, BOX), rng)


if __name__ == '__main__':
    main()
