'''
@author: Michael

Tests for the classes of the project (src/phase/classes/classes.py).
'''

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
import scipy.constants as scp
from scipy.constants import astronomical_unit, hour

from phase.classes import (
    GGN_Noise_Model,
    GravWave,
    GW_strain,
    Interferometer,
    location,
    zeroGW,
)
from phase.decimal_additional_functions import pi

# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Interferometer ~~~
# ~~~~~~~~~~~~~~~~~~~~~~

k_clock = Decimal(2 * scp.pi * 6.8 * 10 ** 9) / Decimal(scp.speed_of_light)  # Single photon k-vector [1/m]


def mzi_settings(**changes: Any) -> dict[str, Any]:
    settings: dict[str, Any] = dict(
        name = "MZI",
        time_intervals = [1, 1],
        upper_momentum = [0, 1, 0, 0],
        lower_momentum = [0, 0, 1, 0],
        upper_internal_state = ["g"] * 4,
        lower_internal_state = ["g"] * 4,
        interaction_type = ["Single_Photon"] * 3,
        detunings = [0, 0, 0],
        k_vector = k_clock
    )
    settings.update(changes)
    return settings


def test_interferometer_defaults():
    interferometer = Interferometer(**mzi_settings())

    assert interferometer.nucleon_number == 87
    assert interferometer.noise is None
    assert interferometer.upper_relaunches is None and interferometer.lower_relaunches is None
    assert float(interferometer.clock_frequency) == pytest.approx(2 * np.pi * 6.8 * 10 ** 9)
    assert interferometer.k_vector == k_clock
    # The lower laser (or retro-reflection mirror) at -1 m and the upper laser one baseline of 12 m above it.
    assert (interferometer.lower_laser_height, interferometer.baseline, interferometer.upper_laser_height) == (-1, 12, Decimal(11))


def test_k_vector_has_no_default():
    settings = mzi_settings()
    del settings["k_vector"]

    with pytest.raises(TypeError, match = "k_vector"):
        Interferometer(**settings)


''' Two-photon pulses need the k-vectors of both beams, [k_1, k_2] with k_eff = k_1 + k_2. '''
@pytest.mark.parametrize("changes", [
    dict(interaction_type = ["Two_Photon"] * 3, k_vector = 2 * k_clock),
    dict(interaction_type = ["Two_Photon"] * 3, k_vector = [k_clock, k_clock, k_clock]),
    dict(k_vector = [k_clock]),
])
def test_interferometer_rejects_wrong_k_vectors(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match = "k_vector = \\[k_1, k_2\\]"):
        Interferometer(**mzi_settings(**changes))


def test_upper_laser_height_is_one_baseline_above_the_lower_laser() -> None:
    interferometer = Interferometer(**mzi_settings(lower_laser_height = Decimal("-2.5"), baseline = 30.25))

    assert interferometer.upper_laser_height == Decimal("27.75")


@pytest.mark.parametrize("baseline", [0, -5])
def test_interferometer_rejects_non_positive_baseline(baseline: int) -> None:
    with pytest.raises(ValueError, match = "baseline"):
        Interferometer(**mzi_settings(baseline = baseline))


''' For P pulses (length of interaction_type): P + 1 momentum and internal state entries per arm, P detunings and P - 1 time intervals between consecutive pulses.
The propagation after the last pulse is the detection_time of numerical_phases, i.e. time_intervals has no entry for it (also no 0). '''
@pytest.mark.parametrize("changes, message", [
    (dict(upper_internal_state = ["g"] * 3), "upper_internal_state has 3"),
    (dict(lower_momentum = [0, 0, 1]), "lower_momentum has 3"),
    (dict(detunings = [0, 0, 0, 0]), "detunings has 4"),
    (dict(time_intervals = [1, 1, 0]), "time_intervals needs 2 entries"),
    (dict(time_intervals = [1]), "time_intervals needs 2 entries"),
])
def test_interferometer_rejects_inconsistent_lists(changes, message):
    with pytest.raises(ValueError, match = message):
        Interferometer(**mzi_settings(**changes))


def clock_mzi(
        upper_momentum: list[int],
        lower_momentum: list[int],
        directions: Sequence[str | None] | None,
        interaction_type: str = "Single_Photon"
    ) -> Interferometer:

    return Interferometer(
        name = "Clock MZI",
        time_intervals = [1, 1],
        upper_momentum = upper_momentum,
        lower_momentum = lower_momentum,
        upper_internal_state = ["g", "e", "g", "g"],
        lower_internal_state = ["g", "g", "e", "g"],
        interaction_type = [interaction_type] * 3,
        detunings = [0, 0, 0],
        k_vector = [k_clock / 2, k_clock / 2] if interaction_type == "Two_Photon" else k_clock,
        single_photon_laser_direction = directions
    )


''' Absorption (g -> e) kicks the atom along the light, stimulated emission (e -> g) against it. A clock MZI that kicks the excited arm up therefore needs light travelling up in every pulse,
the mirrored MZI light travelling down. The direction is case-insensitive. '''
def test_interferometer_accepts_consistent_light_directions():
    assert clock_mzi(upper_momentum = [0, 1, 0, 0], lower_momentum = [0, 0, 1, 0], directions = ["Up"] * 3).single_photon_laser_direction == ["Up"] * 3
    assert clock_mzi(upper_momentum = [0, -1, 0, 0], lower_momentum = [0, 0, -1, 0], directions = ["down", "Down", "DOWN"]).single_photon_laser_direction is not None
    assert clock_mzi(upper_momentum = [0, 1, 0, 0], lower_momentum = [0, 0, 1, 0], directions = [None] * 3, interaction_type = "Two_Photon").single_photon_laser_direction == [None] * 3
    assert clock_mzi(upper_momentum = [0, 1, 0, 0], lower_momentum = [0, 0, 1, 0], directions = None).single_photon_laser_direction is None


@pytest.mark.parametrize("pulse", [0, 1, 2])
def test_interferometer_rejects_light_direction_against_the_kick(pulse):
    directions = ["Up"] * 3
    directions[pulse] = "Down"

    with pytest.raises(ValueError, match = f"single_photon_laser_direction\\[{pulse}\\]"):
        clock_mzi(upper_momentum = [0, 1, 0, 0], lower_momentum = [0, 0, 1, 0], directions = directions)


@pytest.mark.parametrize("directions, interaction_type, message", [
    (["Up"] * 2, "Single_Photon", "one entry per laser pulse"),
    (["Up", "Sideways", "Up"], "Single_Photon", "\"Up\" or \"Down\""),
    (["Up", None, "Up"], "Single_Photon", "\"Up\" or \"Down\""),
    (["Up"] * 3, "Two_Photon", "has to be None"),
])
def test_interferometer_rejects_invalid_light_directions(directions, interaction_type, message):
    with pytest.raises(ValueError, match = message):
        clock_mzi(upper_momentum = [0, 1, 0, 0], lower_momentum = [0, 0, 1, 0], directions = directions, interaction_type = interaction_type)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Gravitational waves ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_zero_gravitational_wave_has_no_strain():
    for time in [0, 0.3, 17]:
        assert GW_strain(gw = zeroGW, time = time) == 0


''' Default phase offset pi/2: s(t) = h cos(2 pi f t). The float phase offset np.pi / 2 limits the precision to ~1e-16 h. '''
def test_gravitational_wave_strain():
    strain = Decimal("1e-20")
    gw = GravWave(frequency = 2, strain = strain)
    tolerance = strain * Decimal("1e-15")

    assert abs(GW_strain(gw = gw, time = 0) - strain) < tolerance
    assert abs(GW_strain(gw = gw, time = Decimal(1) / 8)) < tolerance                   # Quarter period
    assert abs(GW_strain(gw = gw, time = Decimal(1) / 4) + strain) < tolerance          # Half period
    assert abs(GW_strain(gw = gw, time = Decimal("0.13")) - GW_strain(gw = gw, time = Decimal("3.13"))) < tolerance  # Periodicity


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Further containers ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_ggn_noise_model_draws_seed_if_none_given():
    seed = GGN_Noise_Model(type = "NLNM", frequencies = np.array([0.1]), rng = None).rng

    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 32
    assert GGN_Noise_Model(type = "NLNM", frequencies = np.array([0.1]), rng = 17).rng == 17
    assert GGN_Noise_Model().model == "underground"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Sun and Moon positions ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' Hannover at the vernal equinox 2024: The Sun culminates at about 90° - 52.4° = 37.6°. '''
@pytest.mark.slow
def test_sun_and_moon_positions_in_hannover():
    hannover = location(starttime = datetime(2024, 3, 20, 10, 0, 0), location_lon = 9.7, location_lat = 52.4, name = "Hannover", length = 4 * hour, resolution = hour)

    assert len(hannover.alt) == len(hannover.distance) == len(hannover.moonalt) == len(hannover.moondistance) == 4
    assert max(hannover.alt) == pytest.approx(90 - 52.4, abs = 1.5)
    assert all(0.983 < distance / astronomical_unit < 1.017 for distance in hannover.distance)
    assert all(356 * 10 ** 6 < distance < 407 * 10 ** 6 for distance in hannover.moondistance)


def test_decimal_pi_is_used_for_gravitational_waves():
    assert abs(pi() - Decimal("3.14159265358979323846264338327950288419716939937510")) < Decimal("1e-43")


# END OF FILE
