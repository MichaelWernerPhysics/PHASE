from decimal import Decimal

import scipy.constants as scp

from phase.classes import Interferometer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Interferometer Geometries ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" Ready-made interferometer geometries. Each Interferometer is a list of laser pulses: For P pulses, the momentum and internal state lists of both arms have P + 1 entries
(before, between and after the pulses), interaction_type, detunings and single_photon_laser_direction have P entries and time_intervals has P - 1 entries. """

c = Decimal(scp.speed_of_light)


def mach_zehnder(
        T: Decimal | float | int,
        k_vector: Decimal | float | int | list,
        n: int = 1,
        interaction_type: str = "Two_Photon",
        clock: bool = False,
        direction: str = "Up",
        nucleon_number: int = 87,
        clock_frequency: Decimal | float | int | None = None,
        lower_laser_height: Decimal | float | int = -1,
        baseline: Decimal | float | int = 12
    ) -> Interferometer:

    """ Mach-Zehnder interferometer (beam splitter - mirror - beam splitter) with the pulse separation T and n photon momenta per pulse.

    Args:
        T: Time between two consecutive pulses in s.
        k_vector: k-vector in 1/m. Two-photon pulses: the k-vectors of both beams, [k_1, k_2] with k_eff = k_1 + k_2. Single photon pulses: the k-vector of the photon.
        n: Number of photon momenta transferred by each pulse (LMT, imprinted in a single instant).
        interaction_type: "Two_Photon" (e.g. Raman or Bragg) or "Single_Photon" (e.g. a clock transition).
        clock: If True, the kicked arm is in the excited state "e" between the pulses (single photon clock interferometer). Requires include_internal_state_changes in numerical_phases.
        direction: Propagation direction of the light for single photon pulses, "Up" (emitted by the lower laser) or "Down" (emitted by the upper laser).
            The absorption g -> e kicks the atom along the light, s.t. the momentum transfer points in this direction.
        nucleon_number: Mass of the atom in proton masses. Default: Rubidium-87.
        clock_frequency: Angular frequency of the clock transition in rad/s. For single photon pulses with the finite speed of light, it has to equal k_eff c.
            None keeps the default of the Interferometer class.
        lower_laser_height: Height of the lower laser (or retro-reflection mirror) in m.
        baseline: Distance between the lower and the upper laser in m.
    """

    if direction not in ("Up", "Down"):
        raise ValueError(f"The light direction has to be 'Up' or 'Down' (given: {direction!r}).")

    sign = 1 if direction == "Up" else -1
    single_photon = interaction_type == "Single_Photon"
    optional: dict = {} if clock_frequency is None else dict(clock_frequency = clock_frequency)

    return Interferometer(
        name = "MZI",
        time_intervals = [T, T],
        upper_momentum = [0, sign * n, 0, 0],
        lower_momentum = [0, 0, sign * n, 0],
        upper_internal_state = ["g", "e", "g", "g"] if clock else ["g"] * 4,
        lower_internal_state = ["g", "g", "e", "g"] if clock else ["g"] * 4,
        interaction_type = [interaction_type] * 3,
        detunings = [0, 0, 0],
        k_vector = k_vector,
        nucleon_number = nucleon_number,
        lower_laser_height = lower_laser_height,
        baseline = baseline,
        single_photon_laser_direction = [direction if single_photon else None] * 3,
        **optional
    )


def create_MZI(
        time: Decimal | float | int,
        diamonds: int,
        N_LMT: int,
        k_eff: int | float | Decimal,
        lower_laser_height: Decimal | float | int = -1,
        baseline: Decimal | float | int = 12,
        direction: str = "Up"
    ) -> Interferometer:

    """Creates a Mach-Zehnder type interferometer with time T, Q diamonds and periodic internal state changes, driven by single photon pulses.
    LMT is (here) included by imprinted n times the momentum in a single instant. It is left for future work to model LMT via individual light pulses, mainly because of its numerical cost.

    Args:
        time (Decimal | float | int): Time T between the (main) interferometry pulses.
        diamonds (int): Number of diamonds in the interferometer geometry.
        N_LMT (int): Number of Large Momentum Transfer (LMT) pulses per beam splitter pulse. As indicated above: This is not modeled via individual light pulses, but rather by imprinted n times the momentum in a single instant.
        k_eff (int | float | Decimal): k-vector of the single photon pulses. It also sets the clock frequency k_eff c.
        lower_laser_height (Decimal | float | int): Height of the lower laser in m.
        baseline (Decimal | float | int): Distance between the lower and the upper laser in m.
        direction (str): Propagation direction of the light, "Up" (emitted by the lower laser, default) or "Down" (emitted by the upper laser).
            The absorption g -> e kicks the atom along the light and the stimulated emission e -> g against it. With "Down", the momentum transfers therefore point down.
            The laser at the top in Werner, Alibabaei and Gaaloul, "Terrestrial gravitational wave detection with atom interferometers" corresponds to "Down".
            For the gradiometer of its Fig. 3, both directions give the same magnitude of the gravitational wave response (tests/test_numerical_interferometer.py).
    """

    if direction not in ("Up", "Down"):
        raise ValueError(f"The light direction has to be 'Up' or 'Down' (given: {direction!r}).")

    sign = 1 if direction == "Up" else -1
    number_of_intervals = 2 * diamonds

    # One time interval between two consecutive pulses. The time after the last pulse is the detection_time of numerical_phases.
    time_intervals = [time] * number_of_intervals

    # Desired photon momenta within the interferometry paths
    momentum_quanta_upper_path = [0] + [sign * N_LMT, 0] * diamonds + [0]
    momentum_quanta_lower_path = [0] + [0, sign * N_LMT] * diamonds + [0]

    # Ground or excited state: "g" = ground state, "e" = excited state
    # The first element in the list refers to the atomic state *prior* to the first interferometry laser.
    internal_state_upper_path = ["g"] + ["e", "g"] * diamonds + ["g"]
    internal_state_lower_path = ["g"] + ["g", "e"] * diamonds + ["g"]

    return Interferometer(
        name = "MZI",
        time_intervals = time_intervals,
        upper_momentum = momentum_quanta_upper_path,
        lower_momentum = momentum_quanta_lower_path,
        upper_internal_state = internal_state_upper_path,
        lower_internal_state = internal_state_lower_path,
        interaction_type = ["Single_Photon"] * (number_of_intervals + 1),
        detunings = [0] * (number_of_intervals + 1),
        k_vector = k_eff,
        clock_frequency = Decimal(k_eff) * c,
        lower_laser_height = lower_laser_height,
        baseline = baseline,
        # Light direction of each pulse: The absorption g -> e kicks the atom along the light and the stimulated emission e -> g against it.
        single_photon_laser_direction = [direction] * (number_of_intervals + 1)
    )


# END OF FILE
