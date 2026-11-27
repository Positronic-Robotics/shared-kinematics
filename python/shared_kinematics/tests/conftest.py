from pathlib import Path

import numpy as np
import pytest
import shared_kinematics as sk

REPO = Path(__file__).resolve().parents[3]
FR3_URDF = (REPO / 'data' / 'fr3.urdf').read_text()


@pytest.fixture(scope='session')
def fr3() -> sk.ArmModel:
    return sk.load_arm(FR3_URDF, 'fr3', 'robotiq_2f85')


@pytest.fixture(scope='session')
def low() -> np.ndarray:
    """The flange down, the grip point about (0.55, 0, 0.14) m."""
    return np.array([0.0, 0.2, 0.0, -2.2, 0.0, 2.4, np.pi / 4])
