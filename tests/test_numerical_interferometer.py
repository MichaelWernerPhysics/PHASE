'''
@author: Michael

Tests for the numerical interferometer (src/phase/numerical_interferometer.py).

The central check: A Mach-Zehnder interferometer (MZI) in a homogeneous gravitational field has the phase -N k g T^2.
Dividing the numerical phase by -N k T^2 therefore has to give back the local g of the gravitational model, up to small correction terms (finite speed of light, gravity gradients, ...).
'''

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
import scipy.constants as scp

import phase.gravitational_model.gravitational_model as gm
import phase.laser_phase_noise_model.laser_phase_noise_model as lpn
import phase.numerical_interferometer as ni
from phase.classes import (
    GGN_Noise_Model,
    GravWave,
    Interferometer,
    StanfordMassSimplified,
    TestMass as TungstenTestMass,  # Renamed, so that pytest does not try to collect it as a test class.
)
from phase.geometries import create_MZI, mach_zehnder
from phase.gravitational_model import DEFAULT_G, DEFAULT_GRAVITY_GRADIENT
from phase.laser_phase_noise_model import phase_noise
from phase.numerical_interferometer import (
    num_phase_with_origin,
    numerical_differential_interferometer,
    numerical_phases,
    numerical_same_interferometer_height_difference,
)

''' Define some constants. '''
c = Decimal(scp.speed_of_light)
k_clock = Decimal(2 * scp.pi * 6.8 * 10 ** 9) / c        # Single photon k-vector used in the old tests [1/m]
k_raman = Decimal(4 * scp.pi / (780 * 10 ** -9))         # Two-photon Raman k_eff of Rubidium [1/m]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Interferometer Geometries ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def mzi(
        T: Decimal | float | int = 1,
        n: int = 1,
        interaction_type: str = "Single_Photon",
        clock: bool = False,
        nucleon_number: int = 87,
        k_vector: Decimal | list[Decimal] = k_clock,
        lower_laser_height: Decimal | float | int = -1,
        baseline: Decimal | float | int = 12
    ) -> Interferometer:

    """ Mach-Zehnder interferometer with pulse separation T and n photon momenta per beam splitter. If clock = True, the kicked arm is in the excited state.
    Single photon pulses come from the lower laser (light travelling up), s.t. the absorption g -> e kicks the atom up and the stimulated emission e -> g kicks it down.
    The atoms have to stay between the lasers: Tests that launch them far up or down use a larger baseline. """
    return Interferometer(
        name = "MZI",
        time_intervals = [T, T],
        upper_momentum = [0, n, 0, 0],
        lower_momentum = [0, 0, n, 0],
        upper_internal_state = ["g", "e", "g", "g"] if clock else ["g"] * 4,
        lower_internal_state = ["g", "g", "e", "g"] if clock else ["g"] * 4,
        interaction_type = [interaction_type] * 3,
        detunings = [0, 0, 0],
        k_vector = k_vector,
        nucleon_number = nucleon_number,
        lower_laser_height = lower_laser_height,
        baseline = baseline,
        single_photon_laser_direction = ["Up" if interaction_type == "Single_Photon" else None] * 3
    )


def mirrored_clock_mzi(T: Decimal | float | int = 1) -> Interferometer:

    """ Clock MZI with the light coming from the upper laser (travelling down): The absorption g -> e kicks the "upper" arm down, i.e. it is the spatially lower arm. """
    return Interferometer(
        name = "Mirrored clock MZI",
        time_intervals = [T, T],
        upper_momentum = [0, -1, 0, 0],
        lower_momentum = [0, 0, -1, 0],
        upper_internal_state = ["g", "e", "g", "g"],
        lower_internal_state = ["g", "g", "e", "g"],
        interaction_type = ["Single_Photon"] * 3,
        detunings = [0, 0, 0],
        k_vector = k_clock,
        single_photon_laser_direction = ["Down"] * 3
    )


def clock_mzi(T: Decimal | float | int, direction: str) -> Interferometer:
    return mzi(T = T, n = 1, clock = True) if direction == "Up" else mirrored_clock_mzi(T = T)


def swapped_mzi(T: Decimal | float | int = 1) -> Interferometer:

    """ The same MZI as above, but upper and lower arm are exchanged. """
    return Interferometer(
        name = "Swapped MZI",
        time_intervals = [T, T],
        upper_momentum = [0, 0, 1, 0],
        lower_momentum = [0, 1, 0, 0],
        upper_internal_state = ["g"] * 4,
        lower_internal_state = ["g"] * 4,
        interaction_type = ["Single_Photon"] * 3,
        detunings = [0, 0, 0],
        k_vector = k_clock,
        single_photon_laser_direction = ["Up"] * 3
    )


def total_phase(**kwargs: Any) -> Decimal:

    """ Sum of propagation, separation and kick phase. Default: Homogeneous field, no FSL, no Doppler effect, launched with 10 m/s from z = 0. """
    settings: dict[str, Any] = dict(
        initial_height = 0,
        initial_velocity = 10,
        ideal_potential = "homogeneous",
        ideal_trajectory = "homogeneous",
        include_FSL_contribution = False,
        include_Doppler_effect = False,
    )
    settings.update(kwargs)
    phases = numerical_phases(**settings)
    return phases[0] + phases[1] + phases[2]


def g_estimate(
        phase: Decimal,
        n: int,
        k: Decimal,
        T: Decimal | float | int,
        diamonds: int = 1
    ) -> Decimal:

    """ Invert phase = - Q N k g T^2 for g. """
    return - phase / (diamonds * n * k * Decimal(T) ** 2)


def relative_deviation(
        value: Decimal,
        reference: Decimal
    ) -> float:

    return float(abs(value / reference - 1))


# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Helper functions ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

def test_photon_trajectory():
    assert ni.photon_trajectory(start_height = Decimal(1), direction = "up", time_step = Decimal(2)) == Decimal(1) + 2 * c
    assert ni.photon_trajectory(start_height = Decimal(1), direction = "down", time_step = Decimal(2)) == Decimal(1) - 2 * c


def test_photon_trajectory_rejects_unknown_direction():
    with pytest.raises(ValueError):
        ni.photon_trajectory(start_height = Decimal(1), direction = "sideways", time_step = Decimal(2))


@pytest.mark.parametrize("time_step", [Decimal("1e-3"), Decimal("0.5"), Decimal(2)])
def test_atom_trajectory_is_free_fall(time_step: Decimal) -> None:
    z0, v0 = Decimal("1.5"), Decimal(7)
    kwargs: dict[str, Any] = dict(start_height = z0, start_velocity = v0, ideal = "homogeneous", time_step = time_step, absolute_time = Decimal(0), additional_acceleration = Decimal(0))

    assert ni.atom_trajectory(**kwargs) == z0 + v0 * time_step - DEFAULT_G * time_step ** 2 / 2
    assert ni.atom_velocity(**kwargs) == v0 - DEFAULT_G * time_step


def test_photon_travel_time_through_full_baseline():
    travel_time = ni.propagation_time_photon_full_baseline(height_upper = Decimal(11), height_lower = Decimal(-1), time_counter = Decimal(0))

    assert abs(travel_time - Decimal(12) / c) < Decimal("1e-36")


''' A photon meets a falling atom. The interaction time is the root of a quadratic equation, which the bisection-like search in photon_atom_interaction_time has to reproduce down to the smallest time step of 1e-37 s. '''
def test_photon_atom_interaction_time_downwards_photon():
    laser_height, atom_height, atom_velocity = Decimal(11), Decimal(5), Decimal(3)

    # laser_height - c t = atom_height + atom_velocity t - g t^2 / 2
    a, b, c0 = DEFAULT_G / 2, - (c + atom_velocity), laser_height - atom_height
    exact_time = (- b - (b * b - 4 * a * c0).sqrt()) / (2 * a)

    numerical_time = ni.photon_atom_interaction_time(photon_start_height = laser_height, light_direction = "down", target_height = atom_height, target_velocity = atom_velocity, ideal_trajectory = "homogeneous")

    assert abs(numerical_time - exact_time) < Decimal("1e-36")


def test_photon_atom_interaction_time_upwards_photon():
    mirror_height, atom_height, atom_velocity = Decimal(-1), Decimal(5), Decimal(3)

    # mirror_height + c t = atom_height + atom_velocity t - g t^2 / 2
    a, b, c0 = DEFAULT_G / 2, c - atom_velocity, mirror_height - atom_height
    exact_time = (- b + (b * b - 4 * a * c0).sqrt()) / (2 * a)

    numerical_time = ni.photon_atom_interaction_time(photon_start_height = mirror_height, light_direction = "up", target_height = atom_height, target_velocity = atom_velocity, ideal_trajectory = "homogeneous")

    assert abs(numerical_time - exact_time) < Decimal("1e-36")


@pytest.mark.parametrize("photon_height, direction", [(Decimal(1), "down"), (Decimal(9), "up")])
def test_photon_atom_interaction_time_raises_if_photon_already_passed(photon_height, direction):
    with pytest.raises(ni.AtomOutsideBaselineError, match = "cannot reach the atom"):
        ni.photon_atom_interaction_time(photon_start_height = photon_height, light_direction = direction, target_height = Decimal(5), target_velocity = Decimal(0), ideal_trajectory = "homogeneous")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Mach-Zehnder Gravimeter: phase = -N k g T^2 ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' Run a MZI as a gravimeter and estimate the value of g from phase / (-N k T^2).
Without FSL and in a homogeneous field, the result has to be exact (up to Decimal precision). '''
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("T", [Decimal("0.1"), Decimal("0.5"), Decimal(1)])
def test_mzi_measures_g(T, n):
    phase = total_phase(interferometer = mzi(T = T, n = n))

    assert relative_deviation(value = g_estimate(phase = phase, n = n, k = k_clock, T = T), reference = DEFAULT_G) < 1e-12


''' Single photon clock MZI with the light travelling in the direction s (+1 = up, -1 = down). The MZI measures - s k g T^2 (1 + O(1/c)).
With the finite speed of light (FSL), the laser phase only depends on the emission times and the signal comes from the different times both arms spend in the excited state.
Analytic result to first order in 1/c with x = s (g T - v0) / c:
- Without FSL: g (1 + 2 x). This is the redshift of the excited state, i.e. its mass defect in the kinetic and potential energy.
- FSL without Doppler effect: g (1 - x). The photon travel times change the time in the excited state by - 3 x.
- FSL with Doppler effect: g (1 + 2 x). The Doppler tuned laser frequency cancels the - 3 x again.
The propagation phase integrates the redshift exactly along each time step, the remaining deviation (~1e-14) is of second order in 1/c. '''
@pytest.mark.parametrize("include_FSL, include_Doppler", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("T, initial_velocity", [
    (Decimal("0.1"), Decimal(10)),
    (Decimal("0.1"), Decimal(0)),
    (Decimal(1), Decimal(10)),
    (Decimal(1), Decimal(14)),
    (Decimal(1), DEFAULT_G),
])
@pytest.mark.parametrize("direction", ["Up", "Down"])
def test_clock_mzi_measures_g(direction: str, T: Decimal, initial_velocity: Decimal, include_FSL: bool, include_Doppler: bool) -> None:
    s = 1 if direction == "Up" else -1
    phase = total_phase(
        interferometer = clock_mzi(T = T, direction = direction),
        initial_velocity = initial_velocity,
        include_FSL_contribution = include_FSL,
        include_Doppler_effect = include_Doppler,
        include_internal_state_changes = True
    )

    x = s * (DEFAULT_G * T - initial_velocity) / c
    expected_g = DEFAULT_G * (1 - x) if include_FSL and not include_Doppler else DEFAULT_G * (1 + 2 * x)

    assert relative_deviation(value = s * g_estimate(phase = phase, n = 1, k = k_clock, T = T), reference = expected_g) < 1e-13


''' Along the light ray, the laser phase s k (z - z_laser) - w t of a single photon pulse only depends on the emission time. With the FSL (and without the Doppler tuning of the laser frequency),
the spatial and temporal kick phase therefore cancel, up to the tiny mass defect of the excited state in the kick velocity. Without the FSL, the kick phase carries the full signal - s k g T^2. '''
@pytest.mark.parametrize("initial_velocity", [Decimal(10), Decimal(12), Decimal(14)])
@pytest.mark.parametrize("direction", ["Up", "Down"])
def test_single_photon_laser_phase_does_not_depend_on_trajectory(direction: str, initial_velocity: Decimal) -> None:
    s = 1 if direction == "Up" else -1
    settings: dict[str, Any] = dict(
        interferometer = clock_mzi(T = 1, direction = direction),
        initial_height = 0,
        initial_velocity = initial_velocity,
        ideal_potential = "homogeneous",
        ideal_trajectory = "homogeneous",
        include_Doppler_effect = False,
        include_internal_state_changes = True
    )

    kick_with_fsl = numerical_phases(include_FSL_contribution = True, **settings)[2]
    kick_without_fsl = numerical_phases(include_FSL_contribution = False, **settings)[2]

    assert abs(kick_with_fsl) < Decimal("1e-9")
    assert relative_deviation(value = kick_without_fsl, reference = - s * k_clock * DEFAULT_G) < 1e-9


''' LMT in a single instant: One photon momentum belongs to the internal state change, the other n - 1 are counter-propagating LMT pairs, whose laser phase k z remains.
Together with the time in the excited state, the MZI measures n k g T^2. Without internal state change, all n photon momenta are LMT pairs. '''
@pytest.mark.parametrize("n, clock", [(3, True), (100, True), (2, False)])
def test_single_photon_lmt_in_a_single_instant_measures_g(n: int, clock: bool) -> None:
    phase = total_phase(
        interferometer = mzi(T = 1, n = n, clock = clock),
        include_FSL_contribution = True,
        include_Doppler_effect = True,
        include_internal_state_changes = clock
    )

    assert relative_deviation(value = g_estimate(phase = phase, n = n, k = k_clock, T = 1), reference = DEFAULT_G) < 1e-8


def test_two_photon_mzi_measures_g():
    phase = total_phase(interferometer = mzi(T = 1, n = 1, interaction_type = "Two_Photon", k_vector = [k_raman / 2, k_raman / 2]))

    assert relative_deviation(value = g_estimate(phase = phase, n = 1, k = k_raman, T = 1), reference = DEFAULT_G) < 1e-12


''' In the two-photon MZI, the pulse reaches the atom with the beam reflected at the lower mirror, i.e. coming from below. Contrary to the single photon clock MZI above, the laser phase k_eff z of the two
counter-propagating beams does not cancel along the light ray and carries the signal. The photon travel times then shift the pulses in time and the MZI measures g (1 - 3 (g T - v) / c),
with the mean velocity v = v0 + v_rec / 2 of both arms after the first beam splitter (v_rec = hbar k / m).
The correction is independent of k. A tiny k (no recoil, no arm separation) checks that both arms are timed equally and no k-independent phase remains. '''
@pytest.mark.parametrize("k", [Decimal("1e-3"), k_clock, k_raman])
@pytest.mark.parametrize("T, initial_velocity", [
    (Decimal("0.1"), Decimal(10)),
    (Decimal("0.1"), Decimal(0)),
    (Decimal(1), Decimal(10)),
    (Decimal(1), DEFAULT_G),
])
def test_two_photon_mzi_measures_g_with_finite_speed_of_light(T, initial_velocity, k):
    phase = total_phase(interferometer = mzi(T = T, n = 1, interaction_type = "Two_Photon", k_vector = [k / 2, k / 2]), initial_velocity = initial_velocity, include_FSL_contribution = True, include_Doppler_effect = True)

    v_rec = Decimal(scp.hbar) * k / (87 * Decimal(scp.proton_mass))
    expected_g = DEFAULT_G * (1 - 3 * (DEFAULT_G * T - initial_velocity - v_rec / 2) / c)

    assert relative_deviation(value = g_estimate(phase = phase, n = 1, k = k, T = T), reference = expected_g) < 1e-12


''' In a homogeneous field, the total phase does not depend on the detection time: The propagation phase accumulated after the last pulse is cancelled by the change of the separation phase.
detection_time = 0 and detection_time > 0 run through different code paths, and both arms have to end at the same time. '''
@pytest.mark.parametrize("interferometer", [
    mzi(T = 1, n = 1, interaction_type = "Single_Photon", clock = True),
    mirrored_clock_mzi(T = 1),
    mzi(T = 1, n = 1, interaction_type = "Two_Photon", k_vector = [k_raman / 2, k_raman / 2]),
])
@pytest.mark.parametrize("detection_time", [Decimal("0.01"), Decimal("0.1")])
def test_phase_does_not_depend_on_detection_time(interferometer: Interferometer, detection_time: Decimal) -> None:
    settings: dict[str, Any] = dict(interferometer = interferometer, include_FSL_contribution = True, include_Doppler_effect = True, include_internal_state_changes = True)

    assert abs(total_phase(detection_time = detection_time, **settings) - total_phase(**settings)) < Decimal("1e-15")


''' Detection: Both arms end detection_time after the interaction of the later arm with the last pulse; the earlier arm catches up and no arm is propagated backwards.
This also holds for detection times shorter than one time window (T / 1000), which were ignored before, and between one time window and one time window plus the
photon travel time, which gave a negative last step before. '''
@pytest.mark.parametrize("detection_time", [Decimal(0), Decimal("1e-4"), Decimal("0.001") + Decimal("1e-8"), Decimal("0.05")])
def test_detection_time_is_counted_from_the_later_arm(detection_time: Decimal) -> None:
    settings: dict[str, Any] = dict(interferometer = mzi(1, 1, "Two_Photon", k_vector = [k_raman / 2, k_raman / 2]), initial_height = 0, initial_velocity = 10)
    reference = numerical_phases(**settings)
    phases = numerical_phases(detection_time = detection_time, **settings)

    later_arm = max(sum(entry[1] for entry in reference[3]), sum(entry[1] for entry in reference[4]))
    for arm in (phases[3], phases[4]):
        assert sum(entry[1] for entry in arm) == later_arm + detection_time
        assert all(entry[1] >= 0 for entry in arm)


def test_negative_detection_time_is_rejected() -> None:
    with pytest.raises(ValueError, match = "detection time"):
        total_phase(interferometer = mzi(), detection_time = Decimal("-0.1"))


''' The propagation after the last pulse (detection_time) uses time windows of the length of the last pulse separation. With unequal pulse separations,
the stored time windows and the trajectory steps used to come from different intervals, s.t. the phase depended on the detection time. '''
@pytest.mark.parametrize("detection_time", [Decimal("0.01"), Decimal("0.1")])
def test_detection_time_with_unequal_pulse_separations(detection_time: Decimal) -> None:
    interferometer = Interferometer(
        name = "Asymmetric MZI",
        time_intervals = [Decimal("0.8"), Decimal("1.2")],
        upper_momentum = [0, 1, 0, 0],
        lower_momentum = [0, 0, 1, 0],
        upper_internal_state = ["g"] * 4,
        lower_internal_state = ["g"] * 4,
        interaction_type = ["Two_Photon"] * 3,
        detunings = [0, 0, 0],
        k_vector = [k_raman / 2, k_raman / 2]
    )
    settings: dict[str, Any] = dict(interferometer = interferometer)

    assert abs(total_phase(detection_time = detection_time, **settings) - total_phase(**settings)) < Decimal("1e-15")


''' Butterfly: pulses at 0, T, 3T and 4T, the arms cross at 2T and close at 4T. To first order, it does not respond to a constant acceleration: The phase vanishes in a homogeneous field.
The pulse separations are unequal, so the time steps have different lengths in the segments. A sum of the Lagrangian at the end points of the steps left 2 k g T^2 / time_discretization
(7.9e4 rad for 1000 steps here), which only cancels between the arms for symmetric geometries such as the Mach-Zehnder interferometer. '''
@pytest.mark.parametrize("time_discretization", [100, 1000])
def test_butterfly_phase_vanishes_in_homogeneous_field(time_discretization: int) -> None:
    T = Decimal("0.5")
    butterfly = Interferometer(
        name = "Butterfly",
        time_intervals = [T, 2 * T, T],
        upper_momentum = [0, 1, 0, 1, 0],
        lower_momentum = [0, 0, 1, 0, 0],
        upper_internal_state = ["g"] * 5,
        lower_internal_state = ["g"] * 5,
        interaction_type = ["Two_Photon"] * 4,
        detunings = [0] * 4,
        k_vector = [k_raman / 2, k_raman / 2]
    )
    # Launched with 2 g T, the atoms rise to 4.9 m and return to z = 0 at 4T.
    phase = total_phase(interferometer = butterfly, initial_velocity = 2 * DEFAULT_G * T, time_discretization = time_discretization)

    assert abs(phase) < Decimal("1e-25")


''' Multi-diamond clock interferometer from the GW analysis: Q diamonds in a homogeneous field give Q times the MZI phase, i.e. phase = -Q N k g T^2.
The launch velocity Q g T makes the fountain symmetric. For N > 1, the LMT happens in a single instant (see above). '''
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("diamonds", [1, 2, 3])
def test_multi_diamond_interferometer_measures_g(diamonds, n):
    T = Decimal("0.2")
    phase = total_phase(
        interferometer = create_MZI(time = T, diamonds = diamonds, N_LMT = n, k_eff = k_raman),
        initial_velocity = diamonds * DEFAULT_G * T,
        include_FSL_contribution = True,
        include_Doppler_effect = True,
        include_internal_state_changes = True
    )

    assert relative_deviation(value = g_estimate(phase = phase, n = n, k = k_raman, T = T, diamonds = diamonds), reference = DEFAULT_G) < 1e-9


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Scaling laws and symmetries of the MZI ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_phase_is_linear_in_k():
    ratio = total_phase(interferometer = mzi(k_vector = 2 * k_clock)) / total_phase(interferometer = mzi())

    assert abs(ratio - 2) < Decimal("1e-12")


@pytest.mark.parametrize("n", [2, 3, 5])
def test_phase_is_linear_in_momentum_quanta(n):
    ratio = total_phase(interferometer = mzi(T = 1, n = n)) / total_phase(interferometer = mzi(T = 1, n = 1))

    assert abs(ratio - n) < Decimal("1e-12")


def test_phase_is_quadratic_in_pulse_separation():
    ratio = total_phase(interferometer = mzi(T = Decimal("0.8"))) / total_phase(interferometer = mzi(T = Decimal("0.4")))

    assert abs(ratio - 4) < Decimal("1e-12")


@pytest.mark.parametrize("initial_height, initial_velocity", [(0, 4), (3, 10), (-2, 15)])
def test_phase_does_not_depend_on_launch_conditions_in_homogeneous_field(initial_height, initial_velocity):
    # Lasers at -20 m and 20 m: The atoms stay inside the baseline for all launches.
    reference = total_phase(interferometer = mzi(lower_laser_height = -20, baseline = 40))
    phase = total_phase(interferometer = mzi(lower_laser_height = -20, baseline = 40), initial_height = initial_height, initial_velocity = initial_velocity)

    assert abs(phase - reference) < Decimal("1e-20")


def test_phase_does_not_depend_on_absolute_time_offset():
    reference = total_phase(interferometer = mzi())
    phase = total_phase(interferometer = mzi(), absolute_time_offset = Decimal("0.37"))

    assert abs(phase - reference) < Decimal("1e-20")


''' With a test mass (tungsten), both arms have to feel its gravity in every propagation step. Exchanging the arms then flips the sign of the phase exactly,
for the detection directly after the last pulse and after a detection time. With the FSL, the last propagation step before a pulse is shorter than a time window
(it has zero length without the FSL), which covers this step as well. '''
@pytest.mark.parametrize("include_FSL", [False, True])
@pytest.mark.parametrize("detection_time", [Decimal(0), Decimal("0.05")])
def test_test_mass_acts_on_both_arms(detection_time: Decimal, include_FSL: bool) -> None:
    settings: dict[str, Any] = dict(detection_time = detection_time, include_FSL_contribution = include_FSL)
    test_mass: dict[str, Any] = dict(test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0), tungsten_positions = [4.0])
    phase = total_phase(interferometer = mzi(), **settings, **test_mass)
    swapped = total_phase(interferometer = swapped_mzi(), **settings, **test_mass)

    assert abs(phase - total_phase(interferometer = mzi(), **settings)) > Decimal("1e-10")
    assert abs(phase + swapped) < Decimal("1e-20")


''' The propagation phase integrates the Lagrangian along each time step (Simpson's rule on the parabola of the step), so the localized signal of a test mass hardly depends on
time_discretization: 1000 and 2000 steps agree to 5e-10. A sum of the Lagrangian at the end points of the steps differed by 0.5 % and needed the extrapolation 2 signal(2 n) - signal(n). '''
def test_test_mass_signal_does_not_depend_on_the_time_step() -> None:
    test_mass: dict[str, Any] = dict(test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0), tungsten_positions = [4.0])

    def signal(time_discretization: int) -> Decimal:
        return total_phase(interferometer = mzi(), time_discretization = time_discretization, **test_mass) - total_phase(interferometer = mzi(), time_discretization = time_discretization)

    assert relative_deviation(value = signal(time_discretization = 1000), reference = signal(time_discretization = 2000)) < 1e-8
    # The default is 1000 time steps.
    assert total_phase(interferometer = mzi(), **test_mass) == total_phase(interferometer = mzi(), time_discretization = 1000, **test_mass)


def test_swapping_the_arms_flips_the_sign():
    assert abs(total_phase(interferometer = mzi()) + total_phase(interferometer = swapped_mzi())) < Decimal("1e-20")


def test_single_and_two_photon_transitions_agree_without_fsl():
    single = total_phase(interferometer = mzi(T = 1, n = 1, interaction_type = "Single_Photon", k_vector = k_raman))
    double = total_phase(interferometer = mzi(T = 1, n = 1, interaction_type = "Two_Photon", k_vector = [k_raman / 2, k_raman / 2]))

    assert abs(single / double - 1) < Decimal("1e-20")


''' In a homogeneous field (and without FSL) the propagation phase and separation phase of a closed MZI vanish. The full phase is the laser (kick) phase. '''
def test_only_kick_phase_survives_in_homogeneous_field():
    propagation, separation, kick, _, _ = numerical_phases(
        interferometer = mzi(),
        initial_height = 0,
        initial_velocity = 10,
        ideal_potential = "homogeneous",
        ideal_trajectory = "homogeneous",
        include_FSL_contribution = False,
        include_Doppler_effect = False
    )

    assert abs(propagation) < Decimal("1e-25")
    assert abs(separation) < Decimal("1e-25")
    assert kick < 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Correction terms ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

''' Gravity gradient Gamma = DEFAULT_GRAVITY_GRADIENT: phase = -k g T^2 + k Gamma T^2 (z0 + (v0 + v_rec / 2) T - 7/12 g T^2) (e.g. Peters et al., Metrologia 38, 25 (2001)).
The formula is first order in Gamma. The terms of order Gamma^2 T^2 make up the remaining relative deviation of ~2e-6. '''
@pytest.mark.parametrize("initial_height, initial_velocity", [(0, 10), (2, 8), (0, 0)])
def test_gravity_gradient_correction(initial_height, initial_velocity):
    T = Decimal(1)
    interferometer = mzi(T = T, lower_laser_height = -20, baseline = 40)  # The dropped atoms (v0 = 0) fall to -19.6 m.
    homogeneous = total_phase(interferometer = interferometer, initial_height = initial_height, initial_velocity = initial_velocity)
    gradient = total_phase(interferometer = interferometer, initial_height = initial_height, initial_velocity = initial_velocity, ideal_potential = "constant gradient", ideal_trajectory = "constant gradient")

    v_rec = Decimal(scp.hbar) * k_clock / (87 * Decimal(scp.proton_mass))
    expected = k_clock * DEFAULT_GRAVITY_GRADIENT * T ** 2 * (Decimal(initial_height) + (Decimal(initial_velocity) + v_rec / 2) * T - Decimal(7) / 12 * DEFAULT_G * T ** 2)

    assert relative_deviation(value = gradient - homogeneous, reference = expected) < 1e-5


''' Equivalence principle: In a homogeneous field the phase must not depend on the mass of the atom. '''
def test_phase_does_not_depend_on_mass_in_homogeneous_field():
    light = total_phase(interferometer = mzi(T = Decimal("0.2"), n = 5, k_vector = k_raman), initial_velocity = 2)
    heavy = total_phase(interferometer = mzi(T = Decimal("0.2"), n = 5, nucleon_number = 87 * 10 ** 6, k_vector = k_raman), initial_velocity = 2)

    assert abs(light - heavy) < Decimal("1e-20")


''' Recoil x gradient: The kicked arm is on average v_rec T / 2 higher, which adds hbar (N k)^2 Gamma T^3 / (2 m) to the gradient correction (Peters et al., Metrologia 38, 25 (2001)).
Without FSL it is the only mass dependent term, i.e. the phase difference between Rb and a practically infinitely heavy atom.
The separation phase has to use the output velocities after the last pulse, also without detection time (otherwise it cancels the term exactly). '''
@pytest.mark.parametrize("detection_time", [Decimal(0), Decimal("0.01")])
def test_recoil_gravity_gradient_term(detection_time: Decimal) -> None:
    T, n, nucleon_number = Decimal("0.2"), 5, 87
    settings: dict[str, Any] = dict(initial_velocity = 2, ideal_potential = "constant gradient", ideal_trajectory = "constant gradient", detection_time = detection_time)

    light = total_phase(interferometer = mzi(T = T, n = n, nucleon_number = nucleon_number, k_vector = k_raman), **settings)
    heavy = total_phase(interferometer = mzi(T = T, n = n, nucleon_number = nucleon_number * 10 ** 6, k_vector = k_raman), **settings)

    expected = Decimal(scp.hbar) * (n * k_raman) ** 2 * DEFAULT_GRAVITY_GRADIENT * T ** 3 / (2 * nucleon_number * Decimal(scp.proton_mass))

    assert relative_deviation(value = light - heavy, reference = expected) < 10 ** -5


''' A constant additional acceleration (e.g. from the Sun and the Moon) has to shift the g estimate by exactly this acceleration. '''
@pytest.mark.parametrize("model", ["homogeneous", "constant gradient"])
def test_additional_acceleration_shifts_g_estimate(model):
    acceleration = Decimal("1e-3")
    without = total_phase(interferometer = mzi(), ideal_potential = model, ideal_trajectory = model)
    with_acceleration = total_phase(interferometer = mzi(), ideal_potential = model, ideal_trajectory = model, add_gravity = acceleration)

    shift = g_estimate(phase = with_acceleration - without, n = 1, k = k_clock, T = 1)

    assert relative_deviation(value = shift, reference = acceleration) < 10 ** -5


''' The test above does not see on which g the trajectory is based: Without FSL, the total MZI phase only depends on the g of the potential, since kick and propagation phase compensate the g of the trajectory.
Hence the trajectory itself: With an additional acceleration a, the atoms fall a (2 T)^2 / 2 further until the end of the MZI. '''
@pytest.mark.parametrize("model", ["homogeneous", "constant gradient"])
def test_additional_acceleration_accelerates_the_atoms(model: str) -> None:
    acceleration, T = Decimal("1e-6"), Decimal(1)
    settings: dict[str, Any] = dict(interferometer = mzi(T = T), initial_height = 0, initial_velocity = 10, ideal_potential = model, ideal_trajectory = model, include_FSL_contribution = False, include_Doppler_effect = False)

    without = numerical_phases(**settings)
    with_acceleration = numerical_phases(add_gravity = acceleration, **settings)

    for arm in (3, 4):
        assert relative_deviation(value = with_acceleration[arm][-1][0] - without[arm][-1][0], reference = - acceleration * (2 * T) ** 2 / 2) < 10 ** -5


''' The noise list knows 'Mirror Jitter' and 'Laser Phase'. Unknown entries, e.g. the removed 'Gravity' noise, the former name 'Seismic' of the mirror jitter or typos,
are rejected instead of being ignored silently. '''
@pytest.mark.parametrize("noise_source", ["Gravity", "Seismic", "mirror jitter"])
def test_unknown_noise_source_is_rejected(noise_source: str) -> None:
    interferometer = mzi()
    interferometer.noise = [noise_source]

    with pytest.raises(ValueError, match = "Unknown noise source"):
        total_phase(interferometer = interferometer)


def test_gravitational_wave_response_is_linear_in_strain():
    settings = dict(interferometer = mzi(clock = True), include_FSL_contribution = True, include_Doppler_effect = True, include_internal_state_changes = True)
    reference = total_phase(**settings)
    response_1 = total_phase(gw = GravWave(frequency = 0.3, strain = 10 ** -12), **settings) - reference
    response_2 = total_phase(gw = GravWave(frequency = 0.3, strain = 2 * 10 ** -12), **settings) - reference

    assert response_1 != 0
    assert abs(response_2 / response_1 - 2) < Decimal("1e-6")


def test_gravitational_wave_without_strain_changes_nothing():
    settings = dict(interferometer = mzi(clock = True), include_FSL_contribution = True, include_Doppler_effect = True, include_internal_state_changes = True)

    assert total_phase(gw = GravWave(frequency = 0.3, strain = 0), **settings) == total_phase(**settings)


''' The GGN phases are drawn from the seed of the GGN_Noise_Model, so the same seed has to give the same noise realization. '''
@pytest.mark.slow
@pytest.mark.parametrize("model, noise_type", [("seismic", "NLNM"), ("atmospheric", "low")])
def test_ggn_noise_realization_is_fixed_by_seed(model, noise_type):
    frequencies = np.linspace(0.1, 1, 10)
    settings = dict(interferometer = mzi(T = Decimal("0.1")), initial_velocity = 1)
    reference = total_phase(**settings)

    noise_1 = total_phase(GGN_noise = GGN_Noise_Model(type = noise_type, frequencies = frequencies, rng = 5, model = model), **settings) - reference
    noise_2 = total_phase(GGN_noise = GGN_Noise_Model(type = noise_type, frequencies = frequencies, rng = 5, model = model), **settings) - reference
    noise_3 = total_phase(GGN_noise = GGN_Noise_Model(type = noise_type, frequencies = frequencies, rng = 6, model = model), **settings) - reference

    assert noise_1 != 0
    assert noise_1 == noise_2
    assert noise_1 != noise_3


# ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Laser phase noise ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~

def laser_noise_mzi(T: Decimal | float | int = 1, n: int = 1) -> Interferometer:

    """ The MZI from above with the laser phase noise switched on. It only contributes if random phases are passed. """
    interferometer = mzi(T = T, n = n)
    interferometer.noise = ["Laser Phase"]
    return interferometer


def random_laser_phases(seed: int) -> list[float]:
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, len(lpn.frequencies_and_noise)).tolist()


''' The laser phase travels with the light, so the atom picks up the laser phase noise of the emission times 0, T and 2T -- not of the (photon travel time later) interaction times. The MZI combines them to n * (phi(0) - 2 phi(T) + phi(2T)). '''
def test_laser_phase_noise_is_imprinted_at_the_emission_times():
    n, T = 3, Decimal(1)
    laser_phases = random_laser_phases(seed = 1)
    settings = dict(interferometer = laser_noise_mzi(T = T, n = n), include_FSL_contribution = True, include_Doppler_effect = True)

    noise = total_phase(random_phases_laser_noise = laser_phases, **settings) - total_phase(**settings)
    expected = n * (phase_noise(phase_noise_model_phases = laser_phases, time = 0) - 2 * phase_noise(phase_noise_model_phases = laser_phases, time = T) + phase_noise(phase_noise_model_phases = laser_phases, time = 2 * T))

    assert expected != 0
    assert abs(noise - expected) < Decimal("1e-20")


''' In the clock MZI, the laser phase is imprinted with + for the absorption and - for the stimulated emission, independent of the light direction: phi(0) - 2 phi(T) + phi(2T) in both cases.
(The gravity signal - s k g T^2 instead flips its sign with the light direction s.) '''
@pytest.mark.parametrize("direction", ["Up", "Down"])
def test_laser_phase_noise_sign_follows_absorption_and_emission(direction: str) -> None:
    T = Decimal(1)
    laser_phases = random_laser_phases(seed = 2)
    interferometer = clock_mzi(T = T, direction = direction)
    interferometer.noise = ["Laser Phase"]
    settings: dict[str, Any] = dict(interferometer = interferometer, include_FSL_contribution = True, include_Doppler_effect = True, include_internal_state_changes = True)

    noise = total_phase(random_phases_laser_noise = laser_phases, **settings) - total_phase(**settings)
    expected = phase_noise(phase_noise_model_phases = laser_phases, time = 0) - 2 * phase_noise(phase_noise_model_phases = laser_phases, time = T) + phase_noise(phase_noise_model_phases = laser_phases, time = 2 * T)

    assert expected != 0
    assert abs(noise - expected) < Decimal("1e-20")


''' Mirror jitter: A displacement x(t) of the retro-reflection mirror changes the path of the retro-reflected beam by 2 x(t), the direct beam is not affected.
The two-photon pulses pick up 2 k_1 x(t) (k_eff x(t) for k_1 = k_2 = k_eff / 2) and the MZI combines them to 2 k_1 (x(0) - 2 x(T) + x(2T)).
Unequal beams show that only the k-vector of the retro-reflected beam (k_1) enters. Single photon pulses do not use the mirror. '''
@pytest.mark.parametrize("k_1, k_2", [(k_raman / 2, k_raman / 2), (k_raman / 4, 3 * k_raman / 4)])
def test_mirror_jitter_is_imprinted_at_the_pulses(k_1: Decimal, k_2: Decimal) -> None:
    T = Decimal(1)
    random_phases = np.random.default_rng(3).uniform(0, 2 * np.pi, len(gm.frequencies_and_amplitudes)).tolist()
    interferometer = mzi(T, 1, "Two_Photon", k_vector = [k_1, k_2])
    interferometer.noise = ["Mirror Jitter"]
    settings: dict[str, Any] = dict(interferometer = interferometer)

    noise = total_phase(random_phases_mirror_jitter = random_phases, **settings) - total_phase(**settings)
    position = [Decimal(gm.create_displacement_noise_at_time(random_phases = random_phases, time = time)) for time in (0, T, 2 * T)]
    expected = 2 * k_1 * (position[0] - 2 * position[1] + position[2])

    assert expected != 0
    assert abs(noise - expected) < Decimal("1e-20")

    single_photon = mzi(T, 1, "Single_Photon")
    single_photon.noise = ["Mirror Jitter"]
    assert total_phase(interferometer = single_photon, random_phases_mirror_jitter = random_phases) == total_phase(interferometer = single_photon)


''' One random phase per line of the noise model: A shorter list would drop lines (laser phase noise) or fail with an IndexError (mirror jitter), a longer one would be cut. '''
@pytest.mark.parametrize("noise_source, argument", [("Laser Phase", "random_phases_laser_noise"), ("Mirror Jitter", "random_phases_mirror_jitter")])
@pytest.mark.parametrize("difference", [-1, 1])
def test_random_phases_need_one_phase_per_line(noise_source: str, argument: str, difference: int) -> None:
    number_of_lines = len(lpn.frequencies_and_noise) if noise_source == "Laser Phase" else len(gm.frequencies_and_amplitudes)
    interferometer = mzi(T = 1, n = 1, interaction_type = "Two_Photon", k_vector = [k_raman / 2, k_raman / 2])
    interferometer.noise = [noise_source]

    with pytest.raises(ValueError, match = f"{argument} needs one random phase per line of the noise model, i.e. {number_of_lines}"):
        total_phase(interferometer = interferometer, **{argument: [0.0] * (number_of_lines + difference)})

    # One phase per line is accepted.
    total_phase(interferometer = interferometer, **{argument: [0.0] * number_of_lines})


''' The number of lines is read at every call: set_white_phase_noise replaces the noise model, and the random phases have to follow it. '''
def test_random_phases_follow_the_number_of_lines_of_the_noise_model() -> None:
    interferometer = laser_noise_mzi()
    try:
        lpn.set_white_phase_noise(number_of_lines = 10)
        total_phase(interferometer = interferometer, random_phases_laser_noise = [0.0] * 10)
        with pytest.raises(ValueError, match = "i.e. 10"):
            total_phase(interferometer = interferometer, random_phases_laser_noise = [0.0] * lpn.DEFAULT_NUMBER_OF_LINES)
    finally:
        lpn.set_white_phase_noise()


''' Random phases of a noise source that is not in the noise list of the interferometer have no effect. A warning points this out. '''
def test_random_phases_without_noise_source_are_ignored_with_a_warning() -> None:
    random_phases = random_laser_phases(seed = 1)

    with pytest.warns(UserWarning, match = "random_phases_laser_noise is ignored"):
        phase = total_phase(interferometer = mzi(), random_phases_laser_noise = random_phases)

    assert phase == total_phase(interferometer = mzi())


''' Both ensembles of a single-photon gradiometer are driven by the same pulses of the same laser, so the laser phase noise cancels in the differential phase. '''
def test_laser_phase_noise_cancels_in_single_photon_gradiometer() -> None:
    settings: dict[str, Any] = dict(
        interferometer = laser_noise_mzi(T = 1, n = 1000),
        height_delta = Decimal(5),
        init_height = Decimal(0),
        init_velocity = Decimal(10),
        ideal_potential = "constant gradient",
        ideal_trajectory = "constant gradient",
    )

    noisy_phase = numerical_same_interferometer_height_difference(random_phases_laser_noise = random_laser_phases(seed = 1), **settings)
    noise_free_phase = numerical_same_interferometer_height_difference(**settings)

    assert abs(noisy_phase - noise_free_phase) < Decimal("1e-20")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Timing of both arms ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def time_in_states(height_list: list) -> tuple[Decimal, Decimal]:

    """ Total time spent in the ground and excited state along one arm. Entries of the height list are [height, time window, internal state]. """
    ground_state_time = sum((entry[1] for entry in height_list if entry[2] == "g"), Decimal(0))
    excited_state_time = sum((entry[1] for entry in height_list if entry[2] == "e"), Decimal(0))
    assert all(entry[2] in ("g", "e") for entry in height_list)

    return ground_state_time, excited_state_time


''' Both arms of the interferometer have to end at the same time, otherwise the separation phase is evaluated at different times.
After the last pulse, the arm that ends earlier is propagated by the difference. In the MZI this is the lower arm, in the swapped MZI the upper arm. '''
@pytest.mark.parametrize("interferometer", [
    mzi(T = 1, n = 1, interaction_type = "Single_Photon", clock = True),
    mzi(T = 1, n = 1, interaction_type = "Two_Photon", clock = True, k_vector = [k_clock / 2, k_clock / 2]),
    swapped_mzi(T = 1),
])
def test_both_arms_last_equally_long(interferometer):
    phases = numerical_phases(interferometer = interferometer, initial_height = 0, initial_velocity = 10, include_internal_state_changes = True)

    ground_upper, excited_upper = time_in_states(height_list = phases[3])
    ground_lower, excited_lower = time_in_states(height_list = phases[4])

    assert abs((ground_upper + excited_upper) - (ground_lower + excited_lower)) < Decimal("1e-30")


''' In the clock MZI, each arm spends one pulse separation T in the excited state (up to photon travel time differences ~ baseline / c). '''
def test_clock_mzi_spends_T_in_excited_state():
    phases = numerical_phases(interferometer = mzi(T = 1, n = 1, clock = True), initial_height = 0, initial_velocity = 10, include_internal_state_changes = True)

    _, excited_upper = time_in_states(height_list = phases[3])
    _, excited_lower = time_in_states(height_list = phases[4])

    assert abs(excited_upper - 1) < Decimal("1e-7")
    assert abs(excited_lower - 1) < Decimal("1e-7")


''' If an atom leaves the baseline between the lasers, the phase has no meaning: With the finite speed of light, the light cannot reach the atom at the next pulse.
Without it, the trajectories are checked. Both cases raise an error, also in the wrappers. '''
@pytest.mark.parametrize("include_FSL", [True, False])
@pytest.mark.parametrize("initial_velocity, side", [(5, "below"), (16, "above")])
def test_raises_if_atom_leaves_the_baseline(include_FSL: bool, initial_velocity: int, side: str) -> None:
    # Launched with 5 m/s, the atom falls to -9.6 m (below the lower mirror at -1 m) before the last pulse. Launched with 16 m/s, it rises to 13.1 m (above the upper laser at 11 m).
    interferometer = mzi(T = 1, n = 1, interaction_type = "Two_Photon", k_vector = [k_clock / 2, k_clock / 2])
    settings: dict[str, Any] = dict(include_FSL_contribution = include_FSL, include_Doppler_effect = include_FSL)

    with pytest.raises(ni.AtomOutsideBaselineError):
        numerical_phases(interferometer = interferometer, initial_height = 0, initial_velocity = initial_velocity, **settings)
    with pytest.raises(ni.AtomOutsideBaselineError):
        num_phase_with_origin(interferometer = interferometer, init_height = 0, init_velocity = initial_velocity, **settings)
    with pytest.raises(ni.AtomOutsideBaselineError):
        numerical_same_interferometer_height_difference(
            interferometer = interferometer,
            height_delta = Decimal("-0.5") if side == "below" else Decimal("0.5"),
            init_height = Decimal(0),
            init_velocity = Decimal(initial_velocity),
            **settings
        )


''' After the last pulse, the atoms are propagated until the detection. They have to stay inside the baseline during this time as well. '''
def test_raises_if_atom_leaves_the_baseline_before_the_detection() -> None:
    # Launched with 10 m/s, the atoms are at 0.4 m after the last pulse at 2 s, and fall below the lower laser at -1 m about 0.14 s later.
    settings: dict[str, Any] = dict(interferometer = mzi(), initial_height = 0, initial_velocity = 10)
    numerical_phases(detection_time = Decimal("0.05"), **settings)

    with pytest.raises(ni.AtomOutsideBaselineError, match = "leaves the baseline"):
        numerical_phases(detection_time = Decimal("0.2"), **settings)


''' The lasers are part of the Interferometer: the lower laser (or retro-reflection mirror) at lower_laser_height and the upper laser one baseline above it.
Light from below cannot reach an atom below the mirror, light from above cannot reach an atom above the upper laser. '''
def test_laser_geometry_of_the_interferometer_is_used() -> None:
    # Launched with 5 m/s, the atom falls to -9.6 m before the last pulse: below the default mirror at -1 m, above a mirror at -20 m.
    beams = [k_raman / 2, k_raman / 2]
    with pytest.raises(ni.AtomOutsideBaselineError):
        numerical_phases(interferometer = mach_zehnder(T = 1, k_vector = beams), initial_height = 0, initial_velocity = 5)
    assert numerical_phases(interferometer = mach_zehnder(T = 1, k_vector = beams, lower_laser_height = -20, baseline = 32), initial_height = 0, initial_velocity = 5)[0] is not None

    # Launched with 10 m/s, the atom is at 5.1 m at the second pulse: below the default upper laser at -1 m + 12 m = 11 m, above an upper laser at -1 m + 5 m = 4 m.
    def clock_mzi_with_light_from_above(baseline: int) -> Interferometer:
        return mach_zehnder(T = 1, k_vector = k_clock, interaction_type = "Single_Photon", clock = True, direction = "Down", baseline = baseline)

    settings: dict[str, Any] = dict(initial_height = 0, initial_velocity = 10, include_internal_state_changes = True)
    assert numerical_phases(interferometer = clock_mzi_with_light_from_above(baseline = 12), **settings)[0] is not None
    with pytest.raises(ni.AtomOutsideBaselineError):
        numerical_phases(interferometer = clock_mzi_with_light_from_above(baseline = 5), **settings)


''' The laser phase refers to the laser that emits the light (the retro-reflection mirror for two-photon pulses), not to the origin z = 0. Shifting the lasers and the atoms together
in a homogeneous field therefore leaves the kick phase unchanged, also with the Doppler tuning, where k changes from pulse to pulse. With k z measured from the origin, the kick phase
of the Raman MZI changed by - k delta_z g^2 T^2 / c^2 (1.2e-7 rad for 7 m). '''
@pytest.mark.parametrize("geometry", [
    lambda lower: mach_zehnder(T = 1, k_vector = [k_raman / 2, k_raman / 2], lower_laser_height = lower),
    lambda lower: mach_zehnder(T = 1, k_vector = k_clock, interaction_type = "Single_Photon", clock = True, direction = "Up", lower_laser_height = lower),
    lambda lower: mach_zehnder(T = 1, k_vector = k_clock, interaction_type = "Single_Photon", clock = True, direction = "Down", lower_laser_height = lower),
    lambda lower: create_MZI(time = Decimal("0.5"), diamonds = 2, N_LMT = 5, k_eff = k_clock, lower_laser_height = lower),
], ids = ["two-photon", "single photon up", "single photon down", "multi-diamond LMT"])
def test_kick_phase_does_not_depend_on_the_origin(geometry: Callable[[Decimal], Interferometer]) -> None:
    shift = Decimal(7)
    settings: dict[str, Any] = dict(initial_velocity = 10, include_FSL_contribution = True, include_Doppler_effect = True, include_internal_state_changes = True)

    kick = numerical_phases(interferometer = geometry(Decimal(-1)), initial_height = 0, **settings)[2]
    shifted_kick = numerical_phases(interferometer = geometry(Decimal(-1) + shift), initial_height = shift, **settings)[2]

    assert abs(shifted_kick - kick) < Decimal("1e-25")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Wrapper functions ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_num_phase_with_origin_returns_requested_contribution() -> None:
    settings: dict[str, Any] = dict(interferometer = mzi(), init_height = 0, init_velocity = 10)
    propagation, separation, kick, upper, lower = numerical_phases(interferometer = mzi(), initial_height = 0, initial_velocity = 10)

    assert num_phase_with_origin(phase_output = "Propagation", **settings) == propagation
    assert num_phase_with_origin(phase_output = "Separation", **settings) == separation
    assert num_phase_with_origin(phase_output = "Kick", **settings) == kick
    assert num_phase_with_origin(phase_output = "All", **settings) == propagation + separation + kick
    assert num_phase_with_origin(phase_output = "Upper", **settings) == upper
    assert num_phase_with_origin(phase_output = "Lower", **settings) == lower


def test_differential_measurement_of_identical_interferometers_vanishes():
    assert numerical_differential_interferometer(interferometer_1 = mzi(), interferometer_2 = mzi(), init_height = Decimal(0), init_velocity = Decimal(10)) == 0


''' Each interferometer carries its own k-vector (e.g. two atomic species): The phases - k g T^2 and - 2 k g T^2 differ by k g T^2. '''
def test_differential_measurement_with_different_k_vectors() -> None:
    differential_phase = numerical_differential_interferometer(
        interferometer_1 = mzi(),
        interferometer_2 = mzi(k_vector = 2 * k_clock),
        init_height = Decimal(0),
        init_velocity = Decimal(10),
        include_FSL_contribution = False,
        include_Doppler_effect = False
    )

    assert relative_deviation(value = differential_phase, reference = k_clock * DEFAULT_G) < 1e-12


''' Both interferometers of a differential measurement are driven by the same lasers. '''
def test_differential_measurement_needs_the_same_laser_geometry() -> None:
    other_baseline = mzi()
    other_baseline.baseline = 20
    other_mirror = mzi()
    other_mirror.lower_laser_height = -2
    settings: dict[str, Any] = dict(init_height = Decimal(0), init_velocity = Decimal(10))

    for other in (other_baseline, other_mirror):
        with pytest.raises(ValueError, match = "same lower_laser_height and baseline"):
            numerical_differential_interferometer(interferometer_1 = mzi(), interferometer_2 = other, **settings)


''' Gradiometer: Two MZIs separated by delta_z measure phase(z0) - phase(z0 + delta_z) = -k Gamma T^2 delta_z. '''
def test_gradiometer_measures_gravity_gradient():
    delta_z, T = Decimal(1), Decimal(1)
    differential_phase = numerical_same_interferometer_height_difference(
        interferometer = mzi(T = T),
        height_delta = delta_z,
        init_height = Decimal(0),
        init_velocity = Decimal(10),
        ideal_potential = "constant gradient",
        ideal_trajectory = "constant gradient",
        include_FSL_contribution = False,
        include_Doppler_effect = False
    )

    assert relative_deviation(value = differential_phase, reference = - k_clock * DEFAULT_GRAVITY_GRADIENT * T ** 2 * delta_z) < 1e-2


def test_gradiometer_in_homogeneous_field_vanishes():
    differential_phase = numerical_same_interferometer_height_difference(
        interferometer = mzi(),
        height_delta = Decimal(1),
        init_height = Decimal(0),
        init_velocity = Decimal(10),
        include_FSL_contribution = False,
        include_Doppler_effect = False
    )

    assert abs(differential_phase) < Decimal("1e-20")


''' The wrappers pass time_discretization on, so that the convergence can be checked for differential measurements as well:
Each wrapper equals the difference of two numerical_phases calls with the same number of time steps. With a test mass, the phase depends (slightly) on it. '''
def test_differential_wrappers_pass_on_time_discretization() -> None:
    test_mass: dict[str, Any] = dict(test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0), tungsten_positions = [4.0])
    no_fsl: dict[str, Any] = dict(include_FSL_contribution = False, include_Doppler_effect = False)
    wrapper_settings: dict[str, Any] = dict(init_height = Decimal(0), init_velocity = Decimal(10), time_discretization = 200, **no_fsl, **test_mass)
    other_k = mzi(k_vector = 2 * k_clock)

    differential_phase = numerical_differential_interferometer(interferometer_1 = mzi(), interferometer_2 = other_k, **wrapper_settings)
    expected_differential_phase = total_phase(interferometer = mzi(), time_discretization = 200, **test_mass) - total_phase(interferometer = other_k, time_discretization = 200, **test_mass)

    gradiometer_phase = numerical_same_interferometer_height_difference(interferometer = mzi(), height_delta = Decimal(1), **wrapper_settings)
    expected_gradiometer_phase = (
        total_phase(interferometer = mzi(), time_discretization = 200, **test_mass)
        - total_phase(interferometer = mzi(), initial_height = 1, time_discretization = 200, **test_mass)
    )

    assert differential_phase == expected_differential_phase
    assert gradiometer_phase == expected_gradiometer_phase
    # The number of time steps matters for this signal, i.e. the comparison above is not passed with the default of 1000 steps.
    assert expected_differential_phase != total_phase(interferometer = mzi(), **test_mass) - total_phase(interferometer = other_k, **test_mass)


''' The gradiometer wrapper passes z_offset on: The GGN model "underground" depends on the depth z_offset - z below the surface. '''
def test_gradiometer_passes_on_z_offset() -> None:
    noise = GGN_Noise_Model(type = "NHNM", frequencies = np.linspace(0.1, 1, 10), rng = 5, model = "underground")
    settings: dict[str, Any] = dict(GGN_noise = noise, z_offset = 20.0, time_discretization = 200)
    wrapper_settings: dict[str, Any] = dict(init_height = Decimal(0), init_velocity = Decimal(10), include_FSL_contribution = False, include_Doppler_effect = False)

    gradiometer_phase = numerical_same_interferometer_height_difference(interferometer = mzi(), height_delta = Decimal(1), **wrapper_settings, **settings)
    expected_phase = total_phase(interferometer = mzi(), **settings) - total_phase(interferometer = mzi(), initial_height = 1, **settings)
    default_z_offset = numerical_same_interferometer_height_difference(interferometer = mzi(), height_delta = Decimal(1), GGN_noise = noise, time_discretization = 200, **wrapper_settings)

    assert gradiometer_phase == expected_phase
    assert gradiometer_phase != default_z_offset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Single photon pulses with the finite speed of light ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fsl_phases(interferometer: Interferometer, **kwargs: Any) -> list:
    settings: dict[str, Any] = dict(interferometer = interferometer, initial_height = 0, initial_velocity = 10, include_FSL_contribution = True, include_internal_state_changes = True)
    settings.update(kwargs)
    return numerical_phases(**settings)


def test_single_photon_pulses_need_light_direction_with_fsl() -> None:
    interferometer = mzi(clock = True)
    interferometer.single_photon_laser_direction = None

    with pytest.raises(ValueError, match = "single_photon_laser_direction"):
        fsl_phases(interferometer = interferometer)

    # Without the FSL, the light direction does not enter.
    assert fsl_phases(interferometer = interferometer, include_FSL_contribution = False)[0] is not None


def test_single_photon_transitions_need_internal_states_with_fsl() -> None:
    with pytest.raises(ValueError, match = "include_internal_state_changes"):
        fsl_phases(interferometer = mzi(clock = True), include_internal_state_changes = False)


def test_single_photon_state_change_needs_momentum_transfer() -> None:
    interferometer = Interferometer(
        name = "Broken clock MZI",
        time_intervals = [1, 1],
        upper_momentum = [0, 1, 0, 0],
        lower_momentum = [0, 0, 1, 0],
        upper_internal_state = ["g", "e", "g", "g"],
        lower_internal_state = ["g", "e", "e", "g"],
        interaction_type = ["Single_Photon"] * 3,
        detunings = [0, 0, 0],
        k_vector = k_clock,
        single_photon_laser_direction = ["Up"] * 3
    )

    with pytest.raises(ValueError, match = "without momentum transfer"):
        fsl_phases(interferometer = interferometer)


def test_single_photon_frequency_has_to_match_clock_frequency() -> None:
    with pytest.raises(ValueError, match = "clock_frequency"):
        fsl_phases(interferometer = mzi(clock = True, k_vector = k_raman))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ g and gravity gradient as parameters ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' The MZI measures the g that is passed to numerical_phases, not the default value. '''
@pytest.mark.parametrize("g", [Decimal("9.81"), Decimal("1.62")])
def test_mzi_measures_the_given_g(g: Decimal) -> None:
    # On the Moon (1.62 m/s^2), the atoms launched with 10 m/s rise to 31 m.
    phase = total_phase(interferometer = mzi(baseline = 40), g = g)

    assert relative_deviation(value = g_estimate(phase = phase, n = 1, k = k_clock, T = 1), reference = g) < 1e-12


''' The gradiometer measures the gravity gradient that is passed to the wrappers: phase difference - k Gamma T^2 delta_z, see test_gradiometer_measures_gravity_gradient. '''
def test_gradiometer_measures_the_given_gradient() -> None:
    T, delta_z, gradient = Decimal(1), Decimal(5), Decimal("1e-5")
    differential_phase = numerical_same_interferometer_height_difference(
        interferometer = mzi(T = T),
        height_delta = delta_z,
        init_height = Decimal(0),
        init_velocity = Decimal(10),
        ideal_potential = "constant gradient",
        ideal_trajectory = "constant gradient",
        include_FSL_contribution = False,
        include_Doppler_effect = False,
        gravity_gradient = gradient,
    )

    assert relative_deviation(value = differential_phase, reference = - k_clock * gradient * T ** 2 * delta_z) < 1e-2


def test_two_photon_pulses_need_two_k_vectors() -> None:
    interferometer = mzi(interaction_type = "Two_Photon", k_vector = [k_raman / 2, k_raman / 2])
    interferometer.k_vector = k_raman  # Changed after the construction, which validates it.

    with pytest.raises(ValueError, match = "k_vector = \\[k_1, k_2\\]"):
        total_phase(interferometer = interferometer)


# END OF FILE
