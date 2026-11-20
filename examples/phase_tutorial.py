"""
PHASE tutorial: Numerical phase shifts of light-pulse atom interferometers.

Run the whole file with `pixi run tutorial`, or cell by cell (# %%) in VS Code or Jupyter.
Each section computes a phase shift numerically with PHASE and compares it with the analytical result.

numerical_phases returns the list
    [propagation phase, separation phase, kick (laser) phase, trajectory of the upper arm, trajectory of the lower arm].
The phases are Decimal numbers in rad with 45 significant digits, the total phase is the sum of the first three entries.
Heights are measured upwards in m, g > 0 points downwards. By default, the lasers (or mirrors) sit at -1 m and 11 m.
"""

# %% 0. Setup

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy.constants as scp

import phase.laser_phase_noise_model.laser_phase_noise_model as laser_noise
from phase import (
    DEFAULT_G,
    DEFAULT_GRAVITY_GRADIENT,
    GGN_Noise_Model,
    GravWave,
    StanfordMassSimplified,
    TestMass,
    create_MZI,
    mach_zehnder,
    numerical_phases,
    numerical_same_interferometer_height_difference,
)
from phase.classes import location
from phase.earth_moon_gravity_model import acceleration as tidal_acceleration

if __name__ == "__main__":
    matplotlib.use("Agg")  # Run as a script: save the figure instead of opening a window.
import matplotlib.pyplot as plt

c = Decimal(scp.speed_of_light)
hbar = Decimal(scp.hbar)
m_Rb = 87 * Decimal(scp.proton_mass)                    # Mass of Rubidium-87 in kg
k_raman = Decimal(4 * scp.pi / (780 * 10 ** -9))        # Two-photon Raman transition of Rb-87: k_eff = 2 x 2 pi / 780 nm
k_Sr = Decimal(2 * scp.pi / (698 * 10 ** -9))           # Single photon clock transition of Sr-87 at 698 nm
raman_beams = [k_raman / 2, k_raman / 2]                # Two-photon pulses need the k-vectors of both beams, k_eff = k_1 + k_2

g = DEFAULT_G                                           # Standard gravity, 9.80665 m/s^2
Gamma = DEFAULT_GRAVITY_GRADIENT                        # Free-air gradient, 3.086e-6 1/s^2

# Instantaneous light pulses without travel time (infinite speed of light), as in most textbook derivations.
no_fsl: dict[str, Any] = dict(include_FSL_contribution = False, include_Doppler_effect = False)
# The full model: finite speed of light (FSL) and Doppler tuned lasers.
fsl: dict[str, Any] = dict(include_FSL_contribution = True, include_Doppler_effect = True)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def total(phases: list) -> Decimal:
    """ Total phase: propagation + separation + kick phase. """
    return phases[0] + phases[1] + phases[2]


def compare(label: str, numerical: Decimal, analytical: Decimal) -> None:
    deviation = abs(numerical / analytical - 1)
    print(f"{label:<48} numerical {float(numerical): .12e} rad   analytical {float(analytical): .12e} rad   rel. deviation {float(deviation):.1e}")


def section(title: str) -> None:
    print(f"\n{'~' * 8} {title} {'~' * 8}")


# %% 1. The Mach-Zehnder gravimeter: phase = -k_eff g T^2

section("1. Mach-Zehnder gravimeter")
""" Three Raman pulses (beam splitter - mirror - beam splitter) at t = 0, T and 2T. The atoms are launched upwards from z = 0 with 10 m/s.
In a homogeneous field, propagation and separation phase vanish and the laser phase carries the full signal - k_eff g T^2. """

T = Decimal(1)
mzi = mach_zehnder(T = T, k_vector = raman_beams)  # Two-photon MZI with one photon momentum per pulse
phases = numerical_phases(interferometer = mzi, initial_height = 0, initial_velocity = 10, **no_fsl)

print(f"propagation phase {float(phases[0]): .3e} rad, separation phase {float(phases[1]): .3e} rad, kick phase {float(phases[2]): .6e} rad")
compare("Raman MZI, T = 1 s", total(phases), - k_raman * g * T ** 2)

""" The phase scales with T^2: A gravimeter with T = 0.1 s has a 100 times smaller phase. """
T_short = Decimal("0.1")
compare("Raman MZI, T = 0.1 s", total(numerical_phases(interferometer = mach_zehnder(T = T_short, k_vector = raman_beams), initial_height = 0, initial_velocity = 10, **no_fsl)), - k_raman * g * T_short ** 2)


# %% 1b. The two arms of the interferometer

""" The last two entries of numerical_phases are the trajectories of both arms: lists of [height, time step, internal state], with the height at the end of each time step.
Both arms start at the launch height (here z = 0) at t = 0. """

figure, axes = plt.subplots()
for arm, color, label in ((phases[3], "black", "upper arm"), (phases[4], "tab:red", "lower arm")):
    times = np.concatenate(([0.0], np.cumsum([float(entry[1]) for entry in arm])))
    heights = [0.0] + [float(entry[0]) for entry in arm]
    axes.plot(times, heights, color = color, label = label)
for pulse in (0, 1, 2):
    axes.axvline(pulse * float(T), color = "gray", linestyle = ":")
axes.set_xlabel("$t$ in s")
axes.set_ylabel("$z$ in m")
axes.legend()
figure.savefig(OUTPUT_DIRECTORY / "mzi_trajectories.png", dpi = 150)
print(f"Trajectories saved to {OUTPUT_DIRECTORY / 'mzi_trajectories.png'}")


# %% 2. Gravity gradient

section("2. Gravity gradient")
""" In the field g(z) = g - Gamma z, the phase gets the correction k Gamma T^2 (z0 + (v0 + v_rec / 2) T - 7/12 g T^2) (Peters et al., Metrologia 38, 25 (2001)),
with the recoil velocity v_rec = hbar k_eff / m. It depends on the launch conditions: The atoms sample the gradient along their trajectory.
The formula is first order in Gamma: The terms of order Gamma^2 T^2 make up the remaining deviation of ~1e-6. """

gradient: dict[str, Any] = dict(gravity_model = "constant gradient")
for z0, v0 in ((0, 10), (2, 12)):
    homogeneous = total(numerical_phases(interferometer = mzi, initial_height = z0, initial_velocity = v0, **no_fsl))
    with_gradient = total(numerical_phases(interferometer = mzi, initial_height = z0, initial_velocity = v0, **no_fsl, **gradient))
    compare(f"gradient term, z0 = {z0} m, v0 = {v0} m/s", with_gradient - homogeneous, k_raman * Gamma * T ** 2 * (z0 + (v0 + hbar * k_raman / m_Rb / 2) * T - Decimal(7) / 12 * g * T ** 2))

""" Recoil in the gradient: The kicked arm is on average v_rec T / 2 higher, which adds hbar (n k)^2 Gamma T^3 / (2 m). It is the only term that depends on the mass of the atom
(equivalence principle): We compare Rb-87 with a practically infinitely heavy atom. """
T_lmt, n = Decimal("0.2"), 5
light = total(numerical_phases(interferometer = mach_zehnder(T = T_lmt, k_vector = raman_beams, n = n), initial_height = 0, initial_velocity = 2, **no_fsl, **gradient))
heavy = total(numerical_phases(interferometer = mach_zehnder(T = T_lmt, k_vector = raman_beams, n = n, nucleon_number = 87 * 10 ** 6), initial_height = 0, initial_velocity = 2, **no_fsl, **gradient))
compare("recoil term, n = 5, T = 0.2 s", light - heavy, hbar * (n * k_raman) ** 2 * Gamma * T_lmt ** 3 / (2 * m_Rb))


# %% 3. Finite speed of light

section("3. Finite speed of light")
""" The light pulses need time to reach the atoms. For the Raman MZI, the pulse arrives with the beam reflected at the lower mirror, and the MZI measures
g (1 - 3 (g T - v) / c) instead of g. The correction is independent of k_eff and of the mirror positions. Here the atoms are dropped (v0 = 0) from z = 0 with the lower mirror at -25 m.
v is the mean velocity of both arms after the first beam splitter, v = v0 + v_rec / 2 with the recoil velocity v_rec = hbar k_eff / m.

The FSL correction is a small effect (1e-7 of the total phase). PHASE resolves it to ~1e-7 (the order of the neglected terms of the formula), independent of the number of time steps:
The propagation phase integrates the Lagrangian along each time step. """

drop_mzi = mach_zehnder(T = T, k_vector = raman_beams, lower_laser_height = -25, baseline = 26)  # Upper laser at -25 m + 26 m = 1 m
drop: dict[str, Any] = dict(interferometer = drop_mzi, initial_height = 0, initial_velocity = 0)
fsl_correction: dict[int, Decimal] = {}
for steps in (1000, 2000):
    fsl_correction[steps] = total(numerical_phases(time_discretization = steps, **drop, **fsl)) - total(numerical_phases(time_discretization = steps, **drop, **no_fsl))

v_rec = hbar * k_raman / m_Rb
expected_correction = 3 * k_raman * g * T ** 2 * (g * T - (0 + v_rec / 2)) / c
compare("FSL correction, 1000 steps", fsl_correction[1000], expected_correction)
compare("FSL correction, 2000 steps", fsl_correction[2000], expected_correction)


# %% 4. Single photon clock interferometer

section("4. Single photon clock interferometer")
""" Single photon transitions (e.g. the 698 nm clock line of strontium) change the internal state: The kicked arm is in the excited state.
The light direction matters: Absorption kicks the atom along the light. With light from below ("Up"), the phase is - k g T^2, with light from above ("Down") + k g T^2.
With FSL and Doppler tuning, the MZI measures g (1 + 2 x) with x = s (g T - v0) / c and s = +1 (Up) or -1 (Down): the gravitational redshift of the excited state.
The redshift term 2 x is about 7e-9 here. The comparison without it shows that PHASE resolves it. """

T_clock, v0_clock = Decimal("0.1"), 0
for direction, s in (("Up", 1), ("Down", -1)):
    clock_mzi = mach_zehnder(T = T_clock, k_vector = k_Sr, interaction_type = "Single_Photon", clock = True, direction = direction, clock_frequency = k_Sr * c)
    phase = total(numerical_phases(interferometer = clock_mzi, initial_height = 0, initial_velocity = v0_clock, include_internal_state_changes = True, **fsl))
    x = s * (g * T_clock - v0_clock) / c
    compare(f"Sr clock MZI, light {direction}", phase, - s * k_Sr * g * (1 + 2 * x) * T_clock ** 2)
    compare(f"Sr clock MZI, light {direction}, no redshift", phase, - s * k_Sr * g * T_clock ** 2)


# %% 5. Large momentum transfer and multi-diamond geometries

section("5. LMT and multi-diamond geometries")
""" create_MZI builds Q diamonds with n photon momenta per pulse (LMT as a single instantaneous kick), driven by single photon pulses.
The phase scales as - Q n k g T^2, up to the corrections of the finite speed of light of order g T / c. The atoms are launched with Q g T, s.t. they stay inside the baseline. """

T_multi = Decimal("0.2")
for diamonds, lmt in ((1, 1), (1, 10), (2, 10), (3, 10)):
    multi_diamond = create_MZI(time = T_multi, diamonds = diamonds, N_LMT = lmt, k_eff = k_Sr)
    phase = total(numerical_phases(interferometer = multi_diamond, initial_height = 0, initial_velocity = diamonds * g * T_multi, include_internal_state_changes = True, **fsl))
    compare(f"Q = {diamonds} diamonds, n = {lmt}", phase, - diamonds * lmt * k_Sr * g * T_multi ** 2)


# %% 6. Gradiometer

section("6. Gradiometer")
""" Two identical MZIs, separated by delta_z, driven by the same pulses. Their phase difference measures the gravity gradient: - k Gamma T^2 delta_z.
g cancels in the difference: The gradiometer gives the same result for a different g. """

delta_z = Decimal(5)
gradiometer: dict[str, Any] = dict(interferometer = mzi, height_delta = delta_z, init_height = Decimal(0), init_velocity = Decimal(10), **no_fsl, **gradient)
differential = numerical_same_interferometer_height_difference(**gradiometer)
compare("gradiometer, delta_z = 5 m", differential, - k_raman * Gamma * T ** 2 * delta_z)

other_g = numerical_same_interferometer_height_difference(g = Decimal("9.81"), **gradiometer)
print(f"Same gradiometer with g = 9.81 m/s^2: the differential phase changes by {float(other_g / differential - 1):.1e} (relative)")


# %% 7. Gravitational waves

section("7. Gravitational waves")
""" A gravitational wave with the strain h(t) = h sin(2 pi f t + phase offset) changes the light travel time along the baseline. The response is linear in the strain. """

clock_mzi = mach_zehnder(T = 1, k_vector = k_Sr, interaction_type = "Single_Photon", clock = True, clock_frequency = k_Sr * c)
clock_settings: dict[str, Any] = dict(interferometer = clock_mzi, initial_height = 0, initial_velocity = 10, include_internal_state_changes = True, **fsl)
reference = total(numerical_phases(**clock_settings))
response_1 = total(numerical_phases(gw = GravWave(frequency = 0.5, strain = 10 ** -15), **clock_settings)) - reference
response_2 = total(numerical_phases(gw = GravWave(frequency = 0.5, strain = 2 * 10 ** -15), **clock_settings)) - reference
print(f"Response to h = 1e-15: {float(response_1):.4e} rad, to h = 2e-15: {float(response_2):.4e} rad, ratio {float(response_2 / response_1):.6f}")


# %% 8. Laser phase noise and the single photon gradiometer

section("8. Laser phase noise")
""" The laser phase noise is white noise (default: 1 µrad/sqrt(Hz) between 0.1 Hz and 100 Hz, see set_white_phase_noise), represented by sinusoids with random phases.
A single MZI picks it up. In a single photon gradiometer, both ensembles see the same laser pulses, so the noise cancels in the difference (Graham et al., PRL 110, 171102 (2013)). """

noisy_mzi = mach_zehnder(T = 1, k_vector = k_Sr, interaction_type = "Single_Photon", clock = True, clock_frequency = k_Sr * c)
noisy_mzi.noise = ["Laser Phase"]
random_phases = np.random.default_rng(seed = 1).uniform(0, 2 * np.pi, len(laser_noise.frequencies_and_noise)).tolist()  # One random phase per noise line

noise_settings: dict[str, Any] = dict(interferometer = noisy_mzi, initial_height = 0, initial_velocity = 10, include_internal_state_changes = True, **fsl)
single_noise = total(numerical_phases(random_phases_laser_noise = random_phases, **noise_settings)) - total(numerical_phases(**noise_settings))

gradiometer_settings: dict[str, Any] = dict(interferometer = noisy_mzi, height_delta = Decimal(5), init_height = Decimal(0), init_velocity = Decimal(10), include_internal_state_changes = True, **fsl)
noisy_gradiometer = numerical_same_interferometer_height_difference(random_phases_laser_noise = random_phases, **gradiometer_settings)
quiet_gradiometer = numerical_same_interferometer_height_difference(**gradiometer_settings)
print(f"Laser phase noise in a single MZI: {float(single_noise): .3e} rad, in the gradiometer: {float(noisy_gradiometer - quiet_gradiometer): .3e} rad")


# %% 9. A test mass close to the trajectory

section("9. Test mass")
""" A 1.25 kg tungsten ring (simplified Stanford geometry) at z = 4 m attracts the atoms in both arms. The signal is local, but 1000 time steps per pulse separation
already resolve it: The propagation phase integrates the Lagrangian along each step, and the result hardly changes with twice as many steps. """

test_mass: dict[str, Any] = dict(test_mass = TestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0), tungsten_positions = [4.0])
signals: dict[int, Decimal] = {}
for steps in (1000, 2000):
    settings: dict[str, Any] = dict(interferometer = mzi, initial_height = 0, initial_velocity = 10, time_discretization = steps, **no_fsl)
    signals[steps] = total(numerical_phases(**settings, **test_mass)) - total(numerical_phases(**settings))
    print(f"test mass signal with {steps} steps per pulse separation: {float(signals[steps]):.9e} rad")
print(f"relative change from 1000 to 2000 steps: {float(signals[2000] / signals[1000] - 1):.1e}")


# %% 10. Tides of the Sun and the Moon

section("10. Tides")
""" The tidal acceleration of the Sun and the Moon changes g by about 1e-6 m/s^2 over a day. earth_moon_gravity_model computes it for a location,
numerical_phases takes it as add_gravity (positive values increase g). The phase shift is - k_eff a T^2. """

hannover = location(starttime = datetime(2026, 3, 20, 0, 0, 0), location_lon = 9.7, location_lat = 52.4, name = "Hannover", length = 24 * 3600, resolution = 3600)
tides = tidal_acceleration(hannover)
print(f"Tidal acceleration in Hannover on 2026-03-20: {tides.min():.3e} to {tides.max():.3e} m/s^2")

largest_tide = Decimal(float(tides.max()))
tidal_phase = total(numerical_phases(interferometer = mzi, initial_height = 0, initial_velocity = 10, add_gravity = largest_tide, **no_fsl)) - total(phases)
compare("phase shift of the largest tide", tidal_phase, - k_raman * largest_tide * T ** 2)


# %% 11. Gravity gradient noise

section("11. Gravity gradient noise")
""" Seismic waves and density fluctuations of the atmosphere change the local gravity field (gravity gradient noise, GGN). A GGN_Noise_Model draws one realization
with random phases from a seed. The models are "underground" (shaft below the surface), "seismic" (above ground) and "atmospheric". """

ggn_frequencies = np.linspace(0.1, 1.0, 10)
for model, noise_type in (("underground", "NHNM"), ("seismic", "NHNM"), ("atmospheric", "high")):
    realizations = []
    for seed in range(3):
        noise = GGN_Noise_Model(type = noise_type, frequencies = ggn_frequencies, rng = seed, model = model)
        realizations.append(float(total(numerical_phases(interferometer = mzi, initial_height = 0, initial_velocity = 10, GGN_noise = noise, **no_fsl)) - total(phases)))
    print(f"GGN {model:<12} ({noise_type}): phase noise of three realizations " + ", ".join(f"{value: .2e}" for value in realizations) + " rad")


# END OF FILE
