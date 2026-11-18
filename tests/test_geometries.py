'''
@author: Michael

Tests for the ready-made interferometer geometries (src/phase/geometries.py).
The phases of these interferometers are tested in test_numerical_interferometer.py.
'''

from collections.abc import Callable
from decimal import Decimal

import pytest
import scipy.constants as scp

from phase.classes import Interferometer
from phase.geometries import c, create_MZI, mach_zehnder

k_eff_Rb = Decimal(2 * scp.pi * 6.8 * 10 ** 9) / c  # [1/m]


@pytest.mark.parametrize("diamonds", [1, 2, 4])
def test_multi_diamond_geometry(diamonds):
    interferometer = create_MZI(time = Decimal("0.3"), diamonds = diamonds, N_LMT = 3, k_eff = k_eff_Rb)

    assert len(interferometer.time_intervals) == 2 * diamonds
    assert len(interferometer.upper_momentum) == len(interferometer.lower_momentum) == 2 * diamonds + 2
    assert len(interferometer.upper_internal_state) == len(interferometer.lower_internal_state) == 2 * diamonds + 2
    assert len(interferometer.interaction_type) == len(interferometer.detunings) == 2 * diamonds + 1
    assert interferometer.upper_momentum[:3] == [0, 3, 0]
    assert interferometer.lower_momentum[:3] == [0, 0, 3]
    assert interferometer.upper_momentum[-1] == interferometer.lower_momentum[-1] == 0
    # The single photon pulses set both the k-vector and the clock frequency k c.
    assert interferometer.k_vector == k_eff_Rb
    assert interferometer.clock_frequency == k_eff_Rb * c


''' The arm with momentum is the arm in the excited state (clock transitions). '''
def test_kicked_arm_is_excited():
    interferometer = create_MZI(time = Decimal("0.3"), diamonds = 3, N_LMT = 1, k_eff = k_eff_Rb)

    for momentum, state in zip(interferometer.upper_momentum + interferometer.lower_momentum, interferometer.upper_internal_state + interferometer.lower_internal_state, strict = True):
        assert (momentum != 0) == (state == "e")


@pytest.mark.parametrize("direction, sign", [("Up", 1), ("Down", -1)])
def test_mach_zehnder_kicks_along_the_light(direction: str, sign: int) -> None:
    interferometer = mach_zehnder(T = 1, k_vector = k_eff_Rb, n = 2, interaction_type = "Single_Photon", clock = True, direction = direction)

    assert interferometer.upper_momentum == [0, 2 * sign, 0, 0]
    assert interferometer.lower_momentum == [0, 0, 2 * sign, 0]
    assert interferometer.single_photon_laser_direction == [direction] * 3


def test_two_photon_mach_zehnder_has_no_light_direction() -> None:
    interferometer = mach_zehnder(T = 1, k_vector = [k_eff_Rb / 2, k_eff_Rb / 2])

    assert interferometer.interaction_type == ["Two_Photon"] * 3
    assert interferometer.single_photon_laser_direction == [None] * 3
    assert interferometer.upper_internal_state == interferometer.lower_internal_state == ["g"] * 4


def test_mach_zehnder_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match = "Up"):
        mach_zehnder(T = 1, k_vector = k_eff_Rb, interaction_type = "Single_Photon", direction = "Sideways")


@pytest.mark.parametrize("geometry", [
    lambda **laser: mach_zehnder(T = 1, k_vector = k_eff_Rb, interaction_type = "Single_Photon", **laser),
    lambda **laser: create_MZI(time = 1, diamonds = 2, N_LMT = 1, k_eff = k_eff_Rb, **laser),
])
def test_geometries_pass_the_laser_geometry(geometry: Callable[..., Interferometer]) -> None:
    assert geometry().upper_laser_height == Decimal(11)
    assert geometry(lower_laser_height = -5, baseline = 20).upper_laser_height == Decimal(15)


# END OF FILE
