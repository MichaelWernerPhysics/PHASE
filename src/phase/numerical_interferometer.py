from decimal import Decimal

import scipy.constants as spc

from phase.classes import (
    GGN_Noise_Model,
    GravWave,
    GW_strain,
    Interferometer,
    TestMass,
    zeroGW,
)
from phase.gravitational_model import (
    DEFAULT_G,
    DEFAULT_GRAVITY_GRADIENT,
    create_displacement_noise_at_time,
    model_g,
    model_phi,
)
from phase.laser_phase_noise_model import phase_noise

# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Numerical Analysis ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# The dt_list stores the time orders of magnitude in the photon propagation algorithm.
# It therefore starts with a time grid of 10 ** -9s and refines itself until 10 ** -37s.
dt_list = [-9, -11, -13, -15, -17, -19, -21, -23, -25, -27, -29, -31, -33, -35, -37]


# Speed of light
c = 299792458


def photon_trajectory(
    start_height: Decimal,
    direction: str,
    time_step: Decimal
) -> Decimal:
    """Photon trajectory function. Calculates the height of a photon, given its initial condition, flight direction and time.

    Args:
        start_height (Decimal): Initial photon height.
        direction (str): Direction of the flight. Either "up" or "down".
        time_step (Decimal): Time step of the photon flight.

    Returns:
        Decimal: Photon height after the time_step.
    """

    if direction == "up":
        direction_value = 1

    elif direction == "down":
        direction_value = -1

    else:
        raise ValueError("Typo in photon_trajectory function.")

    return start_height + Decimal(direction_value) * c * time_step


# Height of the atom after one time step of length time_step: z + v dt - g(z) dt^2 / 2, with the acceleration g(z) of model_g at start_height.
def atom_trajectory(
    start_height: Decimal,
    start_velocity: Decimal,
    ideal: str,
    time_step: Decimal,
    absolute_time: Decimal,
    additional_acceleration: Decimal,
    g: Decimal = DEFAULT_G,
    gravity_gradient: Decimal = DEFAULT_GRAVITY_GRADIENT,
    test_mass: TestMass | None = None,
    tungsten_positions: list | float | None = None,
) -> Decimal:

    return (
        start_height + start_velocity * time_step - Decimal(0.5)
        * model_g(
            h = start_height,
            ideal = ideal,
            absolute_time = absolute_time,
            additional_global_acceleration = additional_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
        * time_step ** 2
    )


# Velocity of the atom after one time step of length time_step: v - g(z) dt, with the acceleration g(z) of model_g at start_height.
def atom_velocity(
    start_height: Decimal,
    start_velocity: Decimal,
    ideal: str,
    time_step: Decimal,
    absolute_time: Decimal,
    additional_acceleration: Decimal,
    g: Decimal = DEFAULT_G,
    gravity_gradient: Decimal = DEFAULT_GRAVITY_GRADIENT,
    test_mass: TestMass | None = None,
    tungsten_positions: list | float | None = None,
) -> Decimal:

    return (
        start_velocity
        - model_g(
            h = start_height,
            ideal = ideal,
            absolute_time = absolute_time,
            additional_global_acceleration = additional_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
        * time_step
    )


def photon_atom_interaction_time(
    photon_start_height: Decimal,
    light_direction: str,
    target_height: Decimal,
    target_velocity: Decimal,
    ideal_trajectory: str,
    absolute_time: Decimal = Decimal(0),
    add_acceleration: Decimal = Decimal(0),
    g: Decimal = DEFAULT_G,
    gravity_gradient: Decimal = DEFAULT_GRAVITY_GRADIENT,
) -> Decimal:

    abs_time = absolute_time
    rel_time = Decimal(0)

    if light_direction == "up":
        """ Sanity check for boundary conditions."""
        if photon_start_height > target_height:
            print("Photon is already above the atom. No interaction possible.")
            print("photon_start_height", photon_start_height)
            print("target_height", target_height)
            return Decimal("nan")

        for index in range(len(dt_list)):
            dt = Decimal(10 ** dt_list[index])

            while (
                photon_trajectory(start_height = photon_start_height, direction = light_direction, time_step = rel_time + dt)
                < target_height + dt * target_velocity
            ):
                abs_time += dt
                rel_time += dt

                new_height = atom_trajectory(
                    start_height = target_height,
                    start_velocity = target_velocity,
                    ideal = ideal_trajectory,
                    time_step = dt,
                    absolute_time = abs_time,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                new_vel = atom_velocity(
                    start_height = target_height,
                    start_velocity = target_velocity,
                    ideal = ideal_trajectory,
                    time_step = dt,
                    absolute_time = abs_time,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                target_height = new_height
                target_velocity = new_vel

    elif light_direction == "down":
        """ Sanity check of boundary conditions."""
        if photon_start_height < target_height:
            print("Photon is already below the atom. No interaction possible.")
            print("photon_start_height", photon_start_height)
            print("target_height", target_height)
            return Decimal("nan")

        for index in range(len(dt_list)):
            dt = Decimal(10 ** dt_list[index])

            while (
                photon_trajectory(start_height = photon_start_height, direction = light_direction, time_step = rel_time + dt)
                > target_height + dt * target_velocity
            ):
                abs_time += dt
                rel_time += dt

                new_height = atom_trajectory(
                    start_height = target_height,
                    start_velocity = target_velocity,
                    ideal = ideal_trajectory,
                    time_step = dt,
                    absolute_time = abs_time,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                new_vel = atom_velocity(
                    start_height = target_height,
                    start_velocity = target_velocity,
                    ideal = ideal_trajectory,
                    time_step = dt,
                    absolute_time = abs_time,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                target_height = new_height
                target_velocity = new_vel

    else:
        raise ValueError("Typo in photon_atom_interaction_time function.")

    return rel_time


def propagation_time_photon_full_baseline(
    height_upper: Decimal, height_lower: Decimal, time_counter: Decimal
) -> Decimal:

    for index in range(len(dt_list)):
        dt = Decimal(10 ** dt_list[index])

        while photon_trajectory(start_height = height_upper, direction = "down", time_step = time_counter + dt) > height_lower:
            time_counter += dt

    return time_counter


def validate_single_photon_pulses_with_fsl(
    interferometer: Interferometer,
    k_effective: Decimal,
    include_internal_state_changes: bool
) -> None:

    """ With the finite speed of light, the laser phase of a single photon transition only depends on the emission time of the pulse: Along the light ray, the atom sees s k (z - z_laser) - w t = - w t_emission (w = k c).
    The kick phase k z and the photon travel time cancel, and the signal comes from the different times both arms spend in the excited state (Graham et al. 2013, Hu et al. 2017).

    LMT in a single instant: If an arm receives N > 1 photon momenta in one single photon pulse, only one photon belongs to the change of the internal state. The other N - 1 photon momenta are treated as LMT pairs of
    counter-propagating photons (absorption from one laser, stimulated emission into the other one). The laser phase k z of these photons does not cancel, s.t. the interferometer measures N k g T^2 in total.
    Without internal state change, all N photon momenta are treated like this.

    The following conditions are checked for every "Single_Photon" pulse:
    1. The light direction of the pulse is known (single_photon_laser_direction of the interferometer).
    2. An arm that changes its internal state receives momentum.
    3. If any single photon pulse changes an internal state, include_internal_state_changes = True and the photon energy matches the internal energy, k_effective c = clock_frequency. """

    single_photon_pulses = [pulse for pulse, interaction in enumerate(interferometer.interaction_type) if interaction == "Single_Photon"]
    if not single_photon_pulses:
        return

    name = interferometer.name
    directions = interferometer.single_photon_laser_direction

    if directions is None:
        raise ValueError(
            f"Interferometer {name}: Single photon pulses with the finite speed of light need the light direction of each pulse. "
            f"Set single_photon_laser_direction, e.g. [\"Up\"] * {len(interferometer.interaction_type)} if all pulses come from the lower laser."
        )

    any_state_change = False
    for pulse in single_photon_pulses:
        for arm, momentum, states in (("upper", interferometer.upper_momentum, interferometer.upper_internal_state), ("lower", interferometer.lower_momentum, interferometer.lower_internal_state)):
            momentum_transfer = momentum[pulse + 1] - momentum[pulse]
            state_change = states[pulse] != states[pulse + 1]
            any_state_change = any_state_change or state_change

            if state_change and momentum_transfer == 0:
                raise ValueError(
                    f"Interferometer {name}: In single photon pulse {pulse}, the {arm} arm goes from {states[pulse]} to {states[pulse + 1]} without momentum transfer. "
                    "A single photon transition always transfers the photon momentum."
                )

    if not any_state_change:
        return

    if not include_internal_state_changes:
        raise ValueError(
            f"Interferometer {name}: Single photon transitions with the finite speed of light need include_internal_state_changes = True. "
            "The laser phase of a single photon transition does not depend on the atomic trajectory, the gravity signal comes from the time in the excited state."
        )

    photon_frequency = k_effective * Decimal(c)
    clock_frequency = Decimal(interferometer.clock_frequency)
    if abs(photon_frequency / clock_frequency - 1) > Decimal("1e-9"):
        raise ValueError(
            f"Interferometer {name}: The single photon frequency k_effective c = {photon_frequency:.6e} rad/s does not match clock_frequency = {clock_frequency:.6e} rad/s."
        )


# Function that calculates the numerical phase shift of an interferometer, given the initial conditions and the interferometer object and some numerical values for the calculation.
def numerical_phases(
    # Interferometer class object of interest.
    interferometer: Interferometer,
    # Height of the atoms at the start of the interferometer sequence at t=0.
    initial_height: Decimal | float | int,
    # Velocity of the atoms at the start of the interferometer sequence at t=0.
    initial_velocity: Decimal | float | int,
    # Gravitational model of the potential (propagation phase). Options: "homogeneous" or "constant gradient", see gravitational_model.
    ideal_potential: str = "Homogeneous",
    # Gravitational model of the atomic trajectories. Options: "homogeneous" or "constant gradient".
    ideal_trajectory: str = "Homogeneous",
    # Time after the last laser pulse until the detection, counted from the interaction of the later arm with the last pulse. Both arms end at this time, the earlier arm catches up.
    # If zero, the detection happens directly after the last pulse interacted with both arms.
    detection_time: Decimal | float | int = Decimal(0),
    # Time offset to let the AIF sequence not start at t=0, but at any other desired time, e.g. for time dependent signals such as gravitational waves or tides.
    absolute_time_offset: Decimal | float | int = Decimal(0),
    # Random phases of the laser phase noise, one per line of the laser phase noise model (see laser_phase_noise_model). Only used if "Laser Phase" is in the noise list of the interferometer.
    # If None, the laser has no phase noise.
    random_phases_laser_noise: list | None = None,
    # Random phases of the mirror jitter (vibrations of the retro-reflection mirror), one per line of the vibration spectrum (see gravitational_model). Only used if "Mirror Jitter" is in the noise list.
    # If None, the mirror does not move.
    random_phases_mirror_jitter: list | None = None,
    # Additional global acceleration in m/s^2, e.g. the tides of the Sun and the Moon (see earth_moon_gravity_model). Positive values increase g.
    add_gravity: Decimal | float | int | None = None,
    # Possibility to insert a gravitational wave in the form of a GravWave class object.
    gw: GravWave = zeroGW,
    # Gravitational acceleration in m/s^2 and vertical gravity gradient in 1/s^2 (the latter only for the model "constant gradient"). Default: standard gravity and the free-air gradient.
    g: Decimal | float | int = DEFAULT_G,
    gravity_gradient: Decimal | float | int = DEFAULT_GRAVITY_GRADIENT,
    # Two Booleans to decide whether we include the FSL contribution and the Doppler effect in the calculation or not. Mostly used for debugging.
    include_FSL_contribution: bool = True,
    include_Doppler_effect: bool = True,
    # Clock Interferometry True or False
    include_internal_state_changes: bool = False,
    # Tungsten test mass type
    test_mass: TestMass | None = None,
    # Tungsten mass positions for the calculation of the gravitational effect from the tungsten masses. This is a list of the heights of the tungsten masses, if they are included in the calculation.
    tungsten_positions: list | float | None = None,
    # If True, only the gravity of the test masses (and the GGN) is taken into account. Used to isolate the test mass signal.
    only_tungsten: bool = False,
    # If True, the trajectories of both arms are plotted.
    print_trajectory: bool = False,
    # Gravity gradient noise in the form of a GGN_Noise_Model class object. If None, there is no GGN.
    GGN_noise: GGN_Noise_Model | None = None,
    # Reference height in m of the GGN model "underground", see gravity_gradient_noise_potential.
    z_offset: float = 10.0,
    # Number of time steps of each free propagation between two pulses. The propagation phase and the trajectories are first order accurate in the time step, see below.
    time_discretization: int = 1000
) -> list:

    """Initialisation: Data format.
    Get the needed parameters from the input of the function.
    First bring needed parameters into the Decimal format."""

    initial_height = Decimal(initial_height)
    initial_velocity = Decimal(initial_velocity)

    interval_length_num = [Decimal(item) for item in interferometer.time_intervals]

    # Laser geometry of the interferometer: the lower laser (or retro-reflection mirror) and the upper laser one baseline above it.
    lower_laser_height = Decimal(interferometer.lower_laser_height)
    upper_laser_height = interferometer.upper_laser_height
    detection_time = Decimal(detection_time)
    if detection_time < 0:
        raise ValueError(f"The detection time is the propagation after the last pulse and cannot be negative (given: {detection_time}).")
    absolute_time_offset = Decimal(absolute_time_offset)
    g = Decimal(g)
    gravity_gradient = Decimal(gravity_gradient)

    """ Number of time segments for each path of free propagation (argument time_discretization).
    Example: T = 1s, then each time step in the free propagation is 1ms, if time_discretization = 1000.
    For most real life scenarios, a time_discretization of 1000 is sufficient. For local (!) gravitational anharmonicities, such as test masses, the error of the phase is of first order
    in the time step: The test mass signal of a gradiometer can be off by a few percent for 1000 steps. Increase it or extrapolate 2 phi(2 n) - phi(n) in this case. """

    """ Extract the effective k-vector for the two cases of a single photon, or a two-photon process."""
    k_vectors = interferometer.k_vector
    if isinstance(k_vectors, list):
        k_vector_1 = Decimal(k_vectors[0])
        k_vector_2 = Decimal(k_vectors[1])
        k_effective = k_vector_1 + k_vector_2
    else:
        k_effective = Decimal(k_vectors)

    if "Two_Photon" in interferometer.interaction_type and not isinstance(k_vectors, list):
        raise ValueError("Two-photon pulses need the k-vectors of both beams in the Interferometer, k_vector = [k_1, k_2] with k_eff = k_1 + k_2.")

    """ Extracting local variables out of the interferometer object. """
    momentum_quanta_upper_path = interferometer.upper_momentum
    momentum_quanta_lower_path = interferometer.lower_momentum
    internal_state_upper_path = interferometer.upper_internal_state
    internal_state_lower_path = interferometer.lower_internal_state
    pulse_detunings = interferometer.detunings
    interaction_type = interferometer.interaction_type
    nucleon_number = interferometer.nucleon_number
    noise_list = interferometer.noise
    clock_freq = interferometer.clock_frequency
    upper_relaunches = interferometer.upper_relaunches
    lower_relaunches = interferometer.lower_relaunches

    """ Propagation direction of the light ("up" or "down") for each single photon pulse, None for two-photon pulses. """
    if interferometer.single_photon_laser_direction is None:
        laser_direction: list[str | None] = [None] * len(interaction_type)
    else:
        laser_direction = [direction.lower() if isinstance(direction, str) else None for direction in interferometer.single_photon_laser_direction]

    if include_FSL_contribution:
        validate_single_photon_pulses_with_fsl(
            interferometer = interferometer,
            k_effective = k_effective,
            include_internal_state_changes = include_internal_state_changes
        )

    hbar = spc.hbar  # Planck's constant over 2 pi in SI units.
    hbar_over_m = Decimal(hbar / (nucleon_number * spc.proton_mass))  # hbar/m in SI units for the atomic species of interest, given the nucleon number.
    recoil_frequency = Decimal(hbar_over_m * k_effective ** 2 / 2)  # Decimal(0)

    """ Including noise sources:
    Noise list convention includes "Mirror Jitter" and "Laser Phase"; fluctuations of the gravitational field are included via GGN_noise.
    So an example noise list would be ["Mirror Jitter"] if we want to include the mirror jitter, but not laser phase noise.
    If the noise list is empty, no noise is included. """
    mirror_jitter_bool = False
    laser_phase_noise_bool = False

    if noise_list is not None:
        unknown_noise_sources = set(noise_list) - {"Mirror Jitter", "Laser Phase"}
        if unknown_noise_sources:
            raise ValueError(
                f"Unknown noise source(s) {sorted(unknown_noise_sources)}. The noise list knows 'Mirror Jitter' and 'Laser Phase'; fluctuations of the gravitational field are included via GGN_noise."
            )

        if "Mirror Jitter" in noise_list:
            mirror_jitter_bool = True

        if "Laser Phase" in noise_list:
            laser_phase_noise_bool = True

    """ Adding a global acceleration, i.e. from the Earth-Sun-Moon system, if needed.
    This acceleration is assumed to be constant on the time scale of the interferometer and spatially universal on the length scale of the interferometer.

    If no values are given, the additional acceleration is set to zero. """
    if add_gravity is None:
        add_acceleration = Decimal(0)
    else:
        add_acceleration = Decimal(add_gravity)

    """ The clock_mass_gain is the increase of the atomic rest mass due to the internal energy of the excited state. """
    clock_mass_gain = Decimal(clock_freq) * hbar_over_m * Decimal(c) ** -2

    """ Here we will list the number of transferred quanta in each pulse sequence.
    First for the upper path. """
    trans_momentum_upper_path = list[Decimal]()
    for index in range(len(momentum_quanta_upper_path) - 1):
        if (
            momentum_quanta_upper_path[index + 1] - momentum_quanta_upper_path[index]
            == 0
        ):
            """ If no momentum is transferred, we append a zero."""
            trans_momentum_upper_path.append(Decimal(0))

        else:
            """If momentum is transferred, we need to multiply it by the (dimensionless) detuning factor. """
            trans_momentum_upper_path.append(
                Decimal(
                    (momentum_quanta_upper_path[index + 1] - momentum_quanta_upper_path[index]) * (1 + pulse_detunings[index])
                )
            )

    """ And then for the lower path. """
    trans_momentum_lower_path = list[Decimal]()
    for index in range(len(momentum_quanta_lower_path) - 1):
        if (
            momentum_quanta_lower_path[index + 1] - momentum_quanta_lower_path[index]
            == 0
        ):
            """ If no momentum is transferred, we append a zero."""
            trans_momentum_lower_path.append(Decimal(0))

        else:
            """If momentum is transferred, we need to multiply it by the (dimensionless) detuning factor. """
            trans_momentum_lower_path.append(
                Decimal(
                    (momentum_quanta_lower_path[index + 1] - momentum_quanta_lower_path[index]) * (1 + pulse_detunings[index])
                )
            )

    """ Laser detuning:
    Note that we have included the pulse detunings into the trans_momentum lists above. The detunings are therefore included via the momentum transfer via these lists.  """

    """ Intermediate function to numerically calculate the velocity kick due to the photon momentum transfer.
    Included effects: First and second order Doppler shift and the clock mass gain. Higher order relativistic corrections are not included. """

    def doppler_and_mass_corrected_kick_velocity(
        vel: Decimal,  # Atomic velocity right before the interaction.
        internal_state: str,  # Internal state of the atom before the interaction to know its mass for the kick velocity.
    ) -> Decimal:

        if not include_Doppler_effect:
            vel = Decimal(0)

        if internal_state == "g":
            return (k_effective * (1 - Decimal(vel ** 2 / (2 * c ** 2))) + recoil_frequency * Decimal(vel / (c ** 2))) * hbar_over_m

        elif internal_state == "e":
            return (k_effective * (1 - Decimal(vel ** 2 / (2 * c ** 2))) + recoil_frequency * Decimal(vel / (c ** 2))) * hbar_over_m / (1 + clock_mass_gain)

        else:
            raise ValueError("Typo internal state.")

    def direction_sign(pulse: int) -> int:

        """ +1 for light travelling up, -1 for light travelling down. Two-photon pulses and single photon pulses without direction (only allowed without the finite speed of light) count as +1. """
        return -1 if interaction_type[pulse] == "Single_Photon" and laser_direction[pulse] == "down" else 1

    def single_photon_temporal_kick_phase(
        arm: str,  # "upper" or "lower"
        pulse: int,  # Index of the laser pulse (negative indices count from the last pulse).
        interaction_height: Decimal,  # Height of the atom at the interaction, as in the spatial kick phase.
        doppler_shift: Decimal,  # Doppler parameter v/c of the atom at the interaction.
    ) -> Decimal:

        """ The light travelling in direction s (+1 = up, -1 = down) has the phase s k z - w t with w = k c (1 + s doppler_shift), s.t. it is resonant with the moving atom.
        The atom picks up +phase for absorption (g -> e) and -phase for stimulated emission (e -> g). Absorption kicks the atom along the light, s.t. a single photon transition with the photon momentum p (in units of ħk)
        imprints p k z - s p w t. The spatial part p k z is contained in the spatial kick phase. Here we return the temporal part with the photon travel time s (z - z_laser) / c instead of the absolute time,
        since the phase of the laser at emission is common to both arms. Along the light ray, both parts cancel up to the Doppler shift: The laser phase only depends on the emission time.
        The travel time is therefore taken from the same height z as the spatial kick phase. A gravitational wave changes the arrival times of the light (and hence the time in the excited state),
        but not the laser phase that the light carries from its emission to the atom.

        Only the one photon that changes the internal state is a single photon transition. Further photon momenta of the pulse (LMT in a single instant) are LMT pairs of counter-propagating photons,
        whose laser phase k z per photon momentum remains, see validate_single_photon_pulses_with_fsl. """

        pulse = pulse % len(interaction_type)

        if arm == "upper":
            trans_momentum, momentum_quanta, internal_states = trans_momentum_upper_path[pulse], momentum_quanta_upper_path, internal_state_upper_path
        else:
            trans_momentum, momentum_quanta, internal_states = trans_momentum_lower_path[pulse], momentum_quanta_lower_path, internal_state_lower_path

        photon_number = abs(momentum_quanta[pulse + 1] - momentum_quanta[pulse])
        if photon_number == 0 or internal_states[pulse] == internal_states[pulse + 1]:
            return Decimal(0)

        if laser_direction[pulse] not in ("up", "down"):
            raise ValueError(f"Single photon pulse {pulse} without light direction (given: {laser_direction[pulse]!r}).")

        s = direction_sign(pulse = pulse)
        single_photon_momentum = trans_momentum / photon_number
        laser_height = lower_laser_height if s == 1 else upper_laser_height
        travel_time = s * (interaction_height - laser_height) / c

        return - s * single_photon_momentum * travel_time * k_effective * (1 + s * doppler_shift) * c

    abs_time_upper = absolute_time_offset
    abs_time_lower = absolute_time_offset

    upper_spatial_kick_phase_list = list()
    upper_temporal_kick_phase_list = list()
    lower_spatial_kick_phase_list = list()
    lower_temporal_kick_phase_list = list()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ First laser propagation ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """ This time counter starts at zero for the photon travel time calculation. It keeps track of the time the photon(s) need to reach the atom. """
    relative_time_counter = Decimal(0)

    """ Emission time of the pulse. The laser phase travels with the light, s.t. the atom picks up the laser phase noise of the emission time and not the one of the interaction time. """
    first_emission_time = abs_time_upper

    if include_FSL_contribution:
        """ Gravitational Wave Analysis:
        The laser heights are altered due to the presence of a gravitational wave. This is essentially the stretch of the baseline. """
        current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)
        current_upper_laser_height = upper_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

        virtual_atomic_height = initial_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

        """ For two-photon processes and single-photon processes with an upward momentum kick, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
        This duration is altered from the gravitational wave and is stored in this variable. """
        time_duration_photon_baseline = propagation_time_photon_full_baseline(
            height_upper = current_upper_laser_height,
            height_lower = current_lower_laser_height,
            time_counter = relative_time_counter,
        )

        # if interaction_type[0] == "Two_Photon" or (interaction_type[0] == "Single_Photon" and trans_momentum_upper_path[0] > 0) or (interaction_type[0] == "Single_Photon" and trans_momentum_lower_path[0] > 0):
        if interaction_type[0] == "Two_Photon":
            """ To calculate the interaction times, one need to know the interaction process.
            If this is a two-photon process, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
            We therefore start by adding the above calculated time duration to both of the time counters. """
            delta_time = propagation_time_photon_full_baseline(
                height_upper = current_upper_laser_height,
                height_lower = current_lower_laser_height,
                time_counter = relative_time_counter,
            )
            relative_time_counter += delta_time
            abs_time_upper += delta_time
            abs_time_lower += delta_time

            """ Then we can calculate the atom's position and velocity at the time when the photon reaches the lower mirror. """
            atom_height = atom_trajectory(
                start_height = virtual_atomic_height,
                start_velocity = initial_velocity,
                ideal = ideal_trajectory,
                time_step = relative_time_counter,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            atom_vel = atom_velocity(
                start_height = virtual_atomic_height,
                start_velocity = initial_velocity,
                ideal = ideal_trajectory,
                time_step = relative_time_counter,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )

            """ We now calculate the new current laser heights, since the gravitational wave might have altered them during the photon travel time.
            This effect is miniscule, but for the sake of completeness we include it here. Note that we only need to recalculate the lower laser height. """
            current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

            """ Now the photon travels from the lower mirror up to the atom, which takes some additional time depending on the atom's position and velocity at this time. """
            delta_time = photon_atom_interaction_time(
                photon_start_height = current_lower_laser_height,
                light_direction = "up",
                target_height = atom_height,
                target_velocity = atom_vel,
                ideal_trajectory = ideal_trajectory,
                absolute_time = abs_time_upper,
                add_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
            )
            if delta_time.is_nan():
                return [None, None, None, None, None, None]

            relative_time_counter += delta_time
            abs_time_upper += delta_time
            abs_time_lower += delta_time

        # elif interaction_type[0] == "Single_Photon" and (trans_momentum_upper_path[0] < 0 or trans_momentum_lower_path[0] < 0):
        elif interaction_type[0] == "Single_Photon":
            """ In a single photon process, the photon travels directly from the upper laser down to the atom or from the lower laser up to the atom, depending on the light direction of this pulse.
            Note that the light direction is not the direction of the atomic kick: Stimulated emission kicks the atom against the light. """

            # No Photon bouncing, the photon only travels down to the atom
            atom_height = virtual_atomic_height
            atom_vel = initial_velocity

            if laser_direction[0] == "down":
                """ The photon travels from the upper laser down to the atom, which takes some time depending on the atom's position and velocity at this time. """
                delta_time = photon_atom_interaction_time(
                    photon_start_height = current_upper_laser_height,
                    light_direction = "down",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                if delta_time.is_nan():
                    return [None, None, None, None, None, None]

                relative_time_counter += delta_time
                abs_time_upper += delta_time
                abs_time_lower += delta_time

            elif laser_direction[0] == "up":
                """ The photon travels from the lower laser up to the atom, which takes some time depending on the atom's position and velocity at this time. """
                delta_time = photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                if delta_time.is_nan():
                    return [None, None, None, None, None, None]

                relative_time_counter += delta_time
                abs_time_upper += delta_time
                abs_time_lower += delta_time

            else:
                raise ValueError("Single photon pulse without light direction. Set single_photon_laser_direction of the interferometer.")

        else:
            """ No interaction happends with the upper path. There is no interaction process happening here. """
            # There must be an error here, when there is no interaction process defined for the first pulse.
            raise ValueError("No interaction process defined for first pulse.")

        """ Now we know that the travel time of the photon(s) in the first beam splitter took relative_time_counter seconds.
        From this we can calculate the atom's position and velocity at the time of the first interaction: """

        first_interaction_height = atom_trajectory(
            start_height = initial_height,
            start_velocity = initial_velocity,
            ideal = ideal_trajectory,
            time_step = relative_time_counter,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )
        first_interaction_velocity = atom_velocity(
            start_height = initial_height,
            start_velocity = initial_velocity,
            ideal = ideal_trajectory,
            time_step = relative_time_counter,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )

    else:
        first_interaction_height = initial_height
        first_interaction_velocity = initial_velocity

    # Doppler parameter
    if include_Doppler_effect:
        doppler_shift = first_interaction_velocity / c

    else:
        doppler_shift = Decimal(0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Mirror Jitter First Beam Splitter ~~~~~
    # ~~~ Laser Phase Noise First Beam Splitter ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    kick_phase_jitter = Decimal(0)
    if mirror_jitter_bool and interaction_type[0] == "Two_Photon":
        jitter = Decimal(
            create_displacement_noise_at_time(
                random_phases = random_phases_mirror_jitter,
                time = abs_time_upper
            )
        )

        # Single photon pulses do not use the retro-reflection mirror: The mirror jitter only enters two-photon pulses.
        k_vector_Doppler_corrected = k_effective * (1 - Decimal(doppler_shift / 2)) + recoil_frequency * Decimal(doppler_shift / c)
        kick_phase_jitter += (2 * trans_momentum_upper_path[0] * k_vector_Doppler_corrected * jitter)
        kick_phase_jitter += (-2 * trans_momentum_lower_path[0] * k_vector_Doppler_corrected * jitter)

    if laser_phase_noise_bool:
        # Two-photon pulses: The laser phase noise is taken at the emission time of the retro-reflected beam. The direct beam is emitted 2 (z_atom - z_mirror) / c later, this delay is neglected.
        kick_phase_jitter += direction_sign(pulse = 0) * trans_momentum_upper_path[0] * phase_noise(
            phase_noise_model_phases = random_phases_laser_noise,
            time = first_emission_time
        )
        kick_phase_jitter += - direction_sign(pulse = 0) * trans_momentum_lower_path[0] * phase_noise(
            phase_noise_model_phases = random_phases_laser_noise,
            time = first_emission_time
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Imprinted Laser Phase ~~~~
    # ~~~ ("Spatial Kick Phase") ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """ The first laser pulse will alter the (initial) velocity of the atoms depending on the path and is stored in the following variables. """
    first_spatial_kick_phase_laser_contribution_Doppler_corrected_upper = (
        trans_momentum_upper_path[0]
        * doppler_and_mass_corrected_kick_velocity(
            vel = first_interaction_velocity,
            internal_state = internal_state_upper_path[1]
        ).normalize()
    )

    first_spatial_kick_phase_laser_contribution_Doppler_corrected_lower = (
        trans_momentum_lower_path[0]
        * doppler_and_mass_corrected_kick_velocity(
            vel = first_interaction_velocity,
            internal_state = internal_state_lower_path[1]
        ).normalize()
    )

    """ The first laser pulse will imprint a spatial phase on the atoms, which is given by Eq. (35). """
    upper_spatial_kick_phase_list.append(
        first_interaction_height.normalize()
        * first_spatial_kick_phase_laser_contribution_Doppler_corrected_upper.normalize()
        / hbar_over_m
    )
    lower_spatial_kick_phase_list.append(
        first_interaction_height.normalize()
        * first_spatial_kick_phase_laser_contribution_Doppler_corrected_lower.normalize()
        / hbar_over_m
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Photon Flight Times ~~~
    # ~~~ Temporal Kick Phase ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if include_FSL_contribution:
        """ Note that not all the following parameters are used for every interaction type. """
        travel_time_laser_1 = relative_time_counter
        travel_time_laser_2_Doppler = (current_upper_laser_height - first_interaction_height) / c
        travel_time_laser_2 = travel_time_laser_1 - travel_time_laser_2_Doppler

        """ We store the temporal kick phase (i.e. the wt part of the phase imprint) for both paths in the following variables. """
        upper_imprinted_temporal_phase = Decimal(0)
        lower_imprinted_temporal_phase = Decimal(0)

        """ Denote the overall minus sign in the "temporal kick phase" here, since the phase of an EM field is given by ±kz - wt with w = |k|c. """
        if interaction_type[0] == "Two_Photon":
            """ In this two-Photon case, both light fields need to be considered. """

            """ Comment on Doppler shift convention: Laser one travels up (after reflection) and laser two travels down.
            The signs for the Doppler effect are therefore positive for laser one and negative for laser two.
            Because laser one has a positive k vector (traveling upwards), atoms with positive velocity are "moving away" from the light field -- hence the atoms see a lower frequency of the light field one.
            The light field is therefore detuned in the opposite direction, i.e. it is tuned with a positive Doppler shift. See Eq. (33) in the manuscript.
            For the second laser it is the other way around. And with an additional global minus sign, because the photon is emitted into the light field, the phase is therefore substracted from the atom.
            If the atom is kicked downwards, this is taken into account by the trans_momentum variables -- emission and absorption then switch roles. """

            # ~~~~~~~~~~~~~~~~~~
            # ~~~ Upper Path ~~~
            # ~~~~~~~~~~~~~~~~~~

            # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
            upper_imprinted_temporal_phase += (
                - trans_momentum_upper_path[0] * travel_time_laser_1 * k_vector_1 * (1 + doppler_shift) * c
            )

            # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
            upper_imprinted_temporal_phase += (trans_momentum_upper_path[0] * travel_time_laser_2 * k_vector_2 * c)
            upper_imprinted_temporal_phase += (trans_momentum_upper_path[0] * travel_time_laser_2_Doppler * k_vector_2 * (1 - doppler_shift) * c)

            # ~~~~~~~~~~~~~~~~~~
            # ~~~ Lower Path ~~~
            # ~~~~~~~~~~~~~~~~~~

            # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
            lower_imprinted_temporal_phase += (- trans_momentum_lower_path[0] * travel_time_laser_1 * k_vector_1 * (1 + doppler_shift) * c)

            # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
            lower_imprinted_temporal_phase += (trans_momentum_lower_path[0] * travel_time_laser_2 * k_vector_2 * c)

            lower_imprinted_temporal_phase += (trans_momentum_lower_path[0] * travel_time_laser_2_Doppler * k_vector_2 * (1 - doppler_shift) * c)

        elif interaction_type[0] == "Single_Photon":
            """ Single photon pulse: Each arm picks up the temporal part of the laser phase of the light travelling in the direction laser_direction[0], see single_photon_temporal_kick_phase.
            Absorption (g -> e) and stimulated emission (e -> g) differ in the sign of the momentum transfer relative to the light direction. """
            upper_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                arm = "upper",
                pulse = 0,
                interaction_height = first_interaction_height,
                doppler_shift = doppler_shift,
            )
            lower_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                arm = "lower",
                pulse = 0,
                interaction_height = first_interaction_height,
                doppler_shift = doppler_shift,
            )

        upper_temporal_kick_phase_list.append(upper_imprinted_temporal_phase)
        lower_temporal_kick_phase_list.append(lower_imprinted_temporal_phase)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ First atom propagation ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """ We now propagate the atoms through the first interferometer path numerically.
    We do so by filling two lists for each path: One for the heights and one for the velocities.
    A. The entries in height lists are given by [height, time_window], where time_window is the numerical length of each time step. By this one can approximate the propagation integral by step functions.
    B. The entries in the velocity lists are given by the velocities at each time step.

    This is done in four steps:
    1. Write down the atomic position/velocity before the first atom-light interaction.
    2. The first initial conditions arise from the light interaction calculated above.
    3. Then, we propagate the atoms through the interferometer path in a while-loop, until we reach a point where only one more "full" time window would fit into the description before the next photons start propagating at a time t = T (i.e. interval_length_num).
    4. Finally, we do one last propagation step to reach the exact end of the interferometer path, to close the propagation integral on this segment exactly at t = T. """

    """ Define the form of the atomic trajectory and velocity lists.
    The velocity lists only store the atomic velocity on each path at each time step.
    The trajectory lists store a three tuple of the form [height, time, internal state] including the initial height of a time interval, the length of the upcoming time interval and the internal state the atom is in during this time. """
    upper_heights_numerics: list[list] = []
    lower_heights_numerics: list[list] = []
    upper_velocities_numerics = list[Decimal]()
    lower_velocities_numerics = list[Decimal]()

    """ Step 1: Attach the intial conditions to these lists.
    The relative time counter comprises the time needed for the photon(s) to reach the atom. In this time step -- prior to the first beam splitter -- the atoms are in no spatial superposition and only divert in the next step. """
    upper_heights_numerics.append([initial_height, relative_time_counter, internal_state_upper_path[0]])

    if upper_relaunches is None:
        upper_velocities_numerics.append(first_interaction_velocity)
    else:
        upper_velocities_numerics.append(first_interaction_velocity + Decimal(upper_relaunches[0]))

    lower_heights_numerics.append([initial_height, relative_time_counter, internal_state_lower_path[0]])

    if lower_relaunches is None:
        lower_velocities_numerics.append(first_interaction_velocity)
    else:
        lower_velocities_numerics.append(first_interaction_velocity + Decimal(lower_relaunches[0]))

    """ The free atomic evolution starts after the first beam splitter pulse.
    We define the time_window for this propagation to be the length of the path segment (interval_length_num) and divide it by the predefined value of the time_discretization. """
    time_window = interval_length_num[0] / time_discretization

    """ Step 2:
    Next one appends the first time step after the beam splitter.
    For this we need to add the recoil velocity (trans_momentum_upper/lower_path[0] * hbar_over_m * k_effective) and include a factor of 1 / (1 + clock_mass_gain) to include the potentially higer mass of the atom if it is initiallized in the excited state.
    With this velocity the atom is propagated through the first time window (exactly as in the while-loop below), so that the atom and the absolute time counter stay in sync. Otherwise each segment is one time window short and the next pulse hits the atom at z(T - dt).
    We also increase the absolute time counter after this step for each path.
    """
    first_interval_length = interval_length_num[0]

    if internal_state_upper_path[1] == "e":
        kick_velocity_upper = upper_velocities_numerics[0] + trans_momentum_upper_path[0] * hbar_over_m * k_effective / (1 + clock_mass_gain)
    else:
        kick_velocity_upper = upper_velocities_numerics[0] + trans_momentum_upper_path[0] * hbar_over_m * k_effective

    upper_heights_numerics.append(
        [
            atom_trajectory(
                start_height = first_interaction_height,
                start_velocity = kick_velocity_upper,
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            ),
            time_window,
            internal_state_upper_path[1],
        ]
    )
    upper_velocities_numerics.append(
        atom_velocity(
            start_height = upper_heights_numerics[-1][0],
            start_velocity = kick_velocity_upper,
            time_step = time_window,
            ideal = ideal_trajectory,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
    )

    abs_time_upper += time_window

    if internal_state_lower_path[1] == "e":
        kick_velocity_lower = lower_velocities_numerics[0] + trans_momentum_lower_path[0] * hbar_over_m * k_effective / (1 + clock_mass_gain)
    else:
        kick_velocity_lower = lower_velocities_numerics[0] + trans_momentum_lower_path[0] * hbar_over_m * k_effective

    lower_heights_numerics.append(
        [
            atom_trajectory(
                start_height = first_interaction_height,
                start_velocity = kick_velocity_lower,
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            ),
            time_window,
            internal_state_lower_path[1],
        ]
    )
    lower_velocities_numerics.append(
        atom_velocity(
            start_height = lower_heights_numerics[-1][0],
            start_velocity = kick_velocity_lower,
            time_step = time_window,
            ideal = ideal_trajectory,
            absolute_time = abs_time_lower,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
    )

    abs_time_lower += time_window

    """Step 3:
    While-loop propagation through full time windows, until only one more full time window would fit before the next laser pulse starts. """
    while abs_time_upper + time_window <= first_interval_length + absolute_time_offset:

        upper_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = upper_heights_numerics[-1][0],
                    start_velocity = upper_velocities_numerics[-1],
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                ),
                time_window,
                internal_state_upper_path[1],
            ]
        )

        upper_velocities_numerics.append(
            atom_velocity(
                start_height = upper_heights_numerics[-1][0],
                start_velocity = upper_velocities_numerics[-1],
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            )
        )
        abs_time_upper += time_window

    while abs_time_lower + time_window <= first_interval_length + absolute_time_offset:
        lower_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = lower_heights_numerics[-1][0],
                    start_velocity = lower_velocities_numerics[-1],
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                ),
                time_window,
                internal_state_lower_path[1],
            ]
        )
        lower_velocities_numerics.append(
            atom_velocity(
                start_height = lower_heights_numerics[-1][0],
                start_velocity = lower_velocities_numerics[-1],
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            )
        )
        abs_time_lower += time_window

    """ Step 4:
    Final propagation step to reach the exact end of the interferometer path at t = T. """
    time_till_next_laser_pulse_upper = (first_interval_length + absolute_time_offset - abs_time_upper)
    time_till_next_laser_pulse_lower = (first_interval_length + absolute_time_offset - abs_time_lower)

    upper_heights_numerics.append(
        [
            atom_trajectory(
                start_height = upper_heights_numerics[-1][0],
                start_velocity = upper_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = time_till_next_laser_pulse_upper,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            ),
            time_till_next_laser_pulse_upper,
            internal_state_upper_path[1],
        ]
    )
    upper_velocities_numerics.append(
        atom_velocity(
            start_height = upper_heights_numerics[-1][0],
            start_velocity = upper_velocities_numerics[-1],
            ideal = ideal_trajectory,
            time_step = time_till_next_laser_pulse_upper,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
    )
    lower_heights_numerics.append(
        [
            atom_trajectory(
                start_height = lower_heights_numerics[-1][0],
                start_velocity = lower_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = time_till_next_laser_pulse_lower,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            ),
            time_till_next_laser_pulse_lower,
            internal_state_lower_path[1],
        ]
    )
    lower_velocities_numerics.append(
        atom_velocity(
            start_height = lower_heights_numerics[-1][0],
            start_velocity = lower_velocities_numerics[-1],
            ideal = ideal_trajectory,
            time_step = time_till_next_laser_pulse_lower,
            absolute_time = abs_time_lower,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass
        )
    )

    """ The absolute time counter is now increased to the end of the first interferometer segment. """
    abs_time_upper += time_till_next_laser_pulse_upper
    abs_time_lower += time_till_next_laser_pulse_lower

    """ Now we loop over all the middle segments:
    First: Photon flight.
    Then: Atomic propagation.

    Just like in the case of the first beam splitter, however, there are now different atom-light interaction times due to the spatial superposition of the atom. """
    for index in range(len(trans_momentum_upper_path) - 2):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~ Next laser propagation: Upper/lower atomic path ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """ For the next pulse we now need two relative time counters -- one for each atomic path. It keeps track of the time the photon(s) need to reach each path. """
        relative_time_counter_upper_path = Decimal(0)
        relative_time_counter_lower_path = Decimal(0)

        """ UPPER PATH CALCULATION """
        atom_height_upper_path_before = upper_heights_numerics[-1][0]
        atom_vel_upper_path_before = upper_velocities_numerics[-1]
        emission_time_upper = abs_time_upper  # Emission time of the pulse, see the first beam splitter.

        if include_FSL_contribution:
            """ Gravitational Wave Analysis:
            The laser heights are altered due to the presence of a gravitational wave. This is essentially the stretch of the baseline. """
            current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)
            current_upper_laser_height = upper_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

            virtual_upper_atomic_height = atom_height_upper_path_before * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

            """ For two-photon processes and single-photon processes with an upward momentum kick, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
            This duration is altered from the gravitational wave and is stored in this variable. """
            time_duration_photon_baseline = propagation_time_photon_full_baseline(
                height_upper = current_upper_laser_height,
                height_lower = current_lower_laser_height,
                time_counter = relative_time_counter_upper_path,
            )

            # if interaction_type[index + 1] == "Two_Photon" or (interaction_type[index + 1] == "Single_Photon" and momentum_quanta_upper_path[index + 1] > 0):
            if interaction_type[index + 1] == "Two_Photon":
                """ To calculate the interaction times, one need to know the interaction process.
                If this is a two-photon process, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
                We therefore start by adding the above calculated time duration to both of the time counters. """
                relative_time_counter_upper_path += time_duration_photon_baseline
                abs_time_upper += time_duration_photon_baseline

                """ Then we can calculate the atom's position and velocity at the time when the photon reaches the lower mirror. """
                atom_height_upper_path = atom_trajectory(
                    start_height = virtual_upper_atomic_height,
                    start_velocity = atom_vel_upper_path_before,
                    ideal = ideal_trajectory,
                    time_step = relative_time_counter_upper_path,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )
                atom_vel_upper_path = atom_velocity(
                    start_height = atom_height_upper_path_before,
                    start_velocity = atom_vel_upper_path_before,
                    ideal = ideal_trajectory,
                    time_step = relative_time_counter_upper_path,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )

                """ We now calculate the new current laser heights, since the gravitational wave might have altered them during the photon travel time.
                This effect is miniscule, but for the sake of completeness we include it here. Note that we only need to recalculate the lower laser height. """
                current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

                """ Now the photon travels from the lower mirror up to the atom, which takes some additional time depending on the atom's position and velocity at this time. """
                relative_time_counter_upper_path += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height_upper_path,
                    target_velocity = atom_vel_upper_path,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                if relative_time_counter_upper_path.is_nan():
                    return [None, None, None, None, None, None]

                abs_time_upper += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height_upper_path,
                    target_velocity = atom_vel_upper_path,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                if abs_time_upper.is_nan():
                    return [None, None, None, None, None, None]

            # elif interaction_type[index + 1] == "Single_Photon" and momentum_quanta_upper_path[index + 1] < 0:
            elif interaction_type[index + 1] == "Single_Photon":
                """ In a single photon process, the photon travels directly from the upper laser down to the atom or from the lower laser up to the atom, depending on the light direction of this pulse.
                Note that the light direction is not the direction of the atomic kick: Stimulated emission kicks the atom against the light. """

                atom_height = virtual_upper_atomic_height
                atom_vel = atom_vel_upper_path_before

                if laser_direction[index + 1] == "down":
                    """ The photon travels from the upper laser down to the atom, which takes some time depending on the atom's position and velocity at this time. """
                    time_delta = photon_atom_interaction_time(
                        photon_start_height = current_upper_laser_height,
                        light_direction = "down",
                        target_height = atom_height,
                        target_velocity = atom_vel,
                        ideal_trajectory = ideal_trajectory,
                        absolute_time = abs_time_upper,
                        add_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                    )

                    if time_delta.is_nan():
                        return [None, None, None, None, None, None]

                    relative_time_counter_upper_path += time_delta
                    abs_time_upper += time_delta

                elif laser_direction[index + 1] == "up":
                    """ The photon travels from the lower laser up to the atom, which takes some time depending on the atom's position and velocity at this time. """
                    time_delta = photon_atom_interaction_time(
                        photon_start_height = current_lower_laser_height,
                        light_direction = "up",
                        target_height = atom_height,
                        target_velocity = atom_vel,
                        ideal_trajectory = ideal_trajectory,
                        absolute_time = abs_time_upper,
                        add_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                    )

                    if time_delta.is_nan():
                        return [None, None, None, None, None, None]

                    relative_time_counter_upper_path += time_delta
                    abs_time_upper += time_delta

                else:
                    raise ValueError("Single photon pulse without light direction. Set single_photon_laser_direction of the interferometer.")

            """ Now we know that the travel time of the photon(s) in the next beam splitter/mirror took relative_time_counter_upper_path seconds.
            From this we can calculate the atom's position and velocity at the time of the second interaction: """

            next_interaction_height_upper = atom_trajectory(
                start_height = atom_height_upper_path_before,
                start_velocity = atom_vel_upper_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_upper_path,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            next_interaction_velocity_upper = atom_velocity(
                start_height = atom_height_upper_path_before,
                start_velocity = atom_vel_upper_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_upper_path,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )

        else:
            """ If FSL is turned off, the interaction position/velocity are just carried through. """
            next_interaction_height_upper = Decimal(atom_height_upper_path_before)
            next_interaction_velocity_upper = Decimal(atom_vel_upper_path_before)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~ Mirror Jitter Next Interaction Upper Path ~~~~~
        # ~~~ Laser Phase Noise Next Interaction Upper Path ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """ Doppler shift parameter used to know how much the light field needs to be Doppler detuned. """
        if include_Doppler_effect:
            doppler_shift_upper = next_interaction_velocity_upper / c

        else:
            """ Set to zero, if Doppler shift is not included at the beginning of the function. """
            doppler_shift_upper = Decimal(0)

        if mirror_jitter_bool and interaction_type[index + 1] == "Two_Photon":
            jitter = Decimal(
                create_displacement_noise_at_time(
                    random_phases = random_phases_mirror_jitter,
                    time = abs_time_upper
                )
            )

            k_vector_Doppler_corrected = k_effective * (
                1 - Decimal(doppler_shift_upper / 2)
            ) + recoil_frequency * Decimal(doppler_shift_upper / c)

            kick_phase_jitter += (2 * trans_momentum_upper_path[index + 1] * k_vector_Doppler_corrected * jitter)

        if laser_phase_noise_bool:
            kick_phase_jitter += direction_sign(pulse = index + 1) * trans_momentum_upper_path[index + 1] * phase_noise(
                phase_noise_model_phases = random_phases_laser_noise,
                time = emission_time_upper
            )

        """
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~~~ Setback everything to calculate the lower path independently now. ~~~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """

        """ LOWER PATH CALCULATION """
        atom_height_lower_path_before = lower_heights_numerics[-1][0]
        atom_vel_lower_path_before = lower_velocities_numerics[-1]
        emission_time_lower = abs_time_lower  # Emission time of the pulse, see the first beam splitter.

        if include_FSL_contribution:
            """ Gravitational Wave Analysis:
            The laser heights are altered due to the presence of a gravitational wave. This is essentially the stretch of the baseline. """
            current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)
            current_upper_laser_height = upper_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

            virtual_lower_atomic_height = atom_height_lower_path_before * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

            """ For two-photon processes and single-photon processes with an upward momentum kick, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
            This duration is altered from the gravitational wave and is stored in this variable. """
            time_duration_photon_baseline = propagation_time_photon_full_baseline(
                height_upper = current_upper_laser_height,
                height_lower = current_lower_laser_height,
                time_counter = relative_time_counter_lower_path,
            )

            # if interaction_type[index + 1] == "Two_Photon" or (interaction_type[index + 1] == "Single_Photon" and momentum_quanta_lower_path[index + 1] > 0):
            if interaction_type[index + 1] == "Two_Photon":
                """ To calculate the interaction times, one need to know the interaction process.
                If this is a two-photon process, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
                We therefore start by adding the above calculated time duration to both of the time counters. """
                relative_time_counter_lower_path += time_duration_photon_baseline
                abs_time_lower += time_duration_photon_baseline

                """ Then we can calculate the atom's position and velocity at the time when the photon reaches the lower mirror. """
                atom_height_lower_path = atom_trajectory(
                    start_height = virtual_lower_atomic_height,
                    start_velocity = atom_vel_lower_path_before,
                    ideal = ideal_trajectory,
                    time_step = relative_time_counter_lower_path,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )
                atom_vel_lower_path = atom_velocity(
                    start_height = atom_height_lower_path_before,
                    start_velocity = atom_vel_lower_path_before,
                    ideal = ideal_trajectory,
                    time_step = relative_time_counter_lower_path,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )

                """ We now calculate the new current laser heights, since the gravitational wave might have altered them during the photon travel time.
                This effect is miniscule, but for the sake of completeness we include it here. Note that we only need to recalculate the lower laser height. """
                current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

                """ Now the photon travels from the lower mirror up to the atom, which takes some additional time depending on the atom's position and velocity at this time. """
                relative_time_counter_lower_path += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height_lower_path,
                    target_velocity = atom_vel_lower_path,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if relative_time_counter_lower_path.is_nan():
                    return [None, None, None, None, None, None]

                abs_time_lower += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height_lower_path,
                    target_velocity = atom_vel_lower_path,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if abs_time_lower.is_nan():
                    return [None, None, None, None, None, None]

            # elif interaction_type[index + 1] == "Single_Photon" and momentum_quanta_upper_path[index + 1] < 0:
            elif interaction_type[index + 1] == "Single_Photon":
                """ In a single photon process, the photon travels directly from the upper laser down to the atom or from the lower laser up to the atom, depending on the light direction of this pulse.
                Note that the light direction is not the direction of the atomic kick: Stimulated emission kicks the atom against the light. """

                atom_height = virtual_lower_atomic_height
                atom_vel = atom_vel_lower_path_before

                if laser_direction[index + 1] == "down":
                    """ The photon travels from the upper laser down to the atom, which takes some time depending on the atom's position and velocity at this time. """
                    time_delta = photon_atom_interaction_time(
                        photon_start_height = current_upper_laser_height,
                        light_direction = "down",
                        target_height = atom_height,
                        target_velocity = atom_vel,
                        ideal_trajectory = ideal_trajectory,
                        absolute_time = abs_time_lower,
                        add_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                    )

                    if time_delta.is_nan():
                        return [None, None, None, None, None, None]

                    relative_time_counter_lower_path += time_delta
                    abs_time_lower += time_delta

                elif laser_direction[index + 1] == "up":
                    """ The photon travels from the lower laser up to the atom, which takes some time depending on the atom's position and velocity at this time. """
                    time_delta = photon_atom_interaction_time(
                        photon_start_height = current_lower_laser_height,
                        light_direction = "up",
                        target_height = atom_height,
                        target_velocity = atom_vel,
                        ideal_trajectory = ideal_trajectory,
                        absolute_time = abs_time_lower,
                        add_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                    )

                    if time_delta.is_nan():
                        return [None, None, None, None, None, None]

                    relative_time_counter_lower_path += time_delta
                    abs_time_lower += time_delta

                else:
                    raise ValueError("Single photon pulse without light direction. Set single_photon_laser_direction of the interferometer.")

            """ Now we know that the travel time of the photon(s) in the second beam splitter/mirror took relative_time_counter_lower_path seconds.
            From this we can calculate the atom's position and velocity at the time of the second interaction: """

            next_interaction_height_lower = atom_trajectory(
                start_height = atom_height_lower_path_before,
                start_velocity = atom_vel_lower_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_lower_path,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            next_interaction_velocity_lower = atom_velocity(
                start_height = atom_height_lower_path_before,
                start_velocity = atom_vel_lower_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_lower_path,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )

        else:
            next_interaction_height_lower = Decimal(atom_height_lower_path_before)
            next_interaction_velocity_lower = Decimal(atom_vel_lower_path_before)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~ Mirror Jitter Next Interaction Upper Path ~~~~~
        # ~~~ Laser Phase Noise Next Interaction Upper Path ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Doppler parameter
        if include_Doppler_effect:
            doppler_shift_lower = next_interaction_velocity_lower / c

        else:
            doppler_shift_lower = Decimal(0)

        if mirror_jitter_bool and interaction_type[index + 1] == "Two_Photon":
            jitter = Decimal(
                create_displacement_noise_at_time(
                    random_phases = random_phases_mirror_jitter,
                    time = abs_time_lower
                )
            )

            k_vector_Doppler_corrected = k_effective * (
                1 - Decimal(doppler_shift_lower / 2)
            ) + recoil_frequency * Decimal(doppler_shift_lower / c)

            kick_phase_jitter += (- 2 * trans_momentum_lower_path[index + 1] * k_vector_Doppler_corrected * jitter)

        if laser_phase_noise_bool:
            kick_phase_jitter += - direction_sign(pulse = index + 1) * trans_momentum_lower_path[index + 1] * phase_noise(
                phase_noise_model_phases = random_phases_laser_noise,
                time = emission_time_lower
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~ Imprinted Laser Phase ~~~~
        # ~~~ ("Spatial Kick Phase") ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """ The next laser pulse will alter the (initial) velocity of the atoms depending on the path and is stored in the following variables. """
        next_spatial_kick_phase_laser_contribution_Doppler_corrected_upper = (
            trans_momentum_upper_path[index + 1]
            * doppler_and_mass_corrected_kick_velocity(
                vel = next_interaction_velocity_upper,
                internal_state = internal_state_upper_path[index + 2]
            ).normalize()
        )

        next_spatial_kick_phase_laser_contribution_Doppler_corrected_lower = (
            trans_momentum_lower_path[index + 1]
            * doppler_and_mass_corrected_kick_velocity(
                vel = next_interaction_velocity_lower,
                internal_state = internal_state_lower_path[index + 2]
            ).normalize()
        )

        """ The next laser pulse will imprint a spatial phase on the atoms, which is given by Eq. (35) """
        upper_spatial_kick_phase_list.append(
            next_interaction_height_upper.normalize()
            * next_spatial_kick_phase_laser_contribution_Doppler_corrected_upper.normalize()
            / hbar_over_m
        )
        lower_spatial_kick_phase_list.append(
            next_interaction_height_lower.normalize()
            * next_spatial_kick_phase_laser_contribution_Doppler_corrected_lower.normalize()
            / hbar_over_m
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~ Photon Flight Times ~~~~~
        # ~~~ ("Temporal Kick Phase") ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        if include_FSL_contribution:
            """ Note that not all the following parameters are used for every interaction type. """
            travel_time_laser_1_upper = relative_time_counter_upper_path
            travel_time_laser_2_Doppler_upper = (
                current_upper_laser_height - next_interaction_height_upper
            ) / c
            travel_time_laser_2_upper = (
                travel_time_laser_1_upper - travel_time_laser_2_Doppler_upper
            )

            travel_time_laser_1_lower = relative_time_counter_lower_path
            travel_time_laser_2_Doppler_lower = (
                (current_upper_laser_height - next_interaction_height_lower) / c
            )  # Note that the upper laser height is used here, since the photon from laser 2 always starts from the upper laser.
            travel_time_laser_2_lower = (
                travel_time_laser_1_lower - travel_time_laser_2_Doppler_lower
            )

            """ We store the temporal kick phase (i.e. the wt part of the phase imprint) for both paths in the following variables. """
            upper_imprinted_temporal_phase = Decimal(0)
            lower_imprinted_temporal_phase = Decimal(0)

            """ Denote the overall minus sign in the "temporal kick phase" here, since the phase of an EM field is given by ±kz - wt with w = |k|c. """
            if interaction_type[index + 1] == "Two_Photon":
                """ In this two-Photon case, both light fields need to be considered. """

                """ Comment on Doppler shift convention: Laser one travels up (after reflection) and laser two travels down.
                The signs for the Doppler effect are therefore positive for laser one and negative for laser two.
                Because laser one has a positive k vector (traveling upwards), atoms with positive velocity are "moving away" from the light field -- hence the atoms see a lower frequency of the light field one.
                The light field is therefore detuned in the opposite direction, i.e. it is tuned with a positive Doppler shift. See Eq. (33) in the manuscript.
                For the second laser it is the other way around. And with an additional global minus sign, because the photon is emitted into the light field, the phase is therefore substracted from the atom.
                If the atom is kicked downwards, this is taken into account by the trans_momentum variables -- emission and absorption then switch roles. """

                # ~~~~~~~~~~~~~~~~~~
                # ~~~ Upper Path ~~~
                # ~~~~~~~~~~~~~~~~~~

                # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
                upper_imprinted_temporal_phase += (
                    - trans_momentum_upper_path[index + 1] * travel_time_laser_1_upper * k_vector_1 * (1 + doppler_shift_upper) * c
                )

                # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
                upper_imprinted_temporal_phase += (
                    trans_momentum_upper_path[index + 1] * travel_time_laser_2_upper * k_vector_2 * c
                )
                upper_imprinted_temporal_phase += (
                    trans_momentum_upper_path[index + 1] * travel_time_laser_2_Doppler_upper * k_vector_2 * (1 - doppler_shift_upper) * c
                )

                # ~~~~~~~~~~~~~~~~~~
                # ~~~ Lower Path ~~~
                # ~~~~~~~~~~~~~~~~~~

                # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
                lower_imprinted_temporal_phase += (
                    - trans_momentum_lower_path[index + 1] * travel_time_laser_1_lower * k_vector_1 * (1 + doppler_shift_lower) * c
                )

                # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
                lower_imprinted_temporal_phase += (
                    trans_momentum_lower_path[index + 1] * travel_time_laser_2_lower * k_vector_2 * c
                )
                lower_imprinted_temporal_phase += (
                    trans_momentum_lower_path[index + 1] * travel_time_laser_2_Doppler_lower * k_vector_2 * (1 - doppler_shift_lower) * c
                )

            elif interaction_type[index + 1] == "Single_Photon":
                """ Single photon pulse: Each arm picks up the temporal part of the laser phase of the light travelling in the direction laser_direction[index + 1], see single_photon_temporal_kick_phase.
                Absorption (g -> e) and stimulated emission (e -> g) differ in the sign of the momentum transfer relative to the light direction. """
                upper_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                    arm = "upper",
                    pulse = index + 1,
                    interaction_height = next_interaction_height_upper,
                    doppler_shift = doppler_shift_upper,
                )
                lower_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                    arm = "lower",
                    pulse = index + 1,
                    interaction_height = next_interaction_height_lower,
                    doppler_shift = doppler_shift_lower,
                )

            """ Append the temporal kick phases to their respective lists. """
            upper_temporal_kick_phase_list.append(upper_imprinted_temporal_phase)
            lower_temporal_kick_phase_list.append(lower_imprinted_temporal_phase)

            """ Append the interaction heights and velocities for the numerical trajectory calculation after the laser pulse. """

            upper_heights_numerics.append(
                [
                    atom_trajectory(
                        start_height = upper_heights_numerics[-1][0],
                        start_velocity = upper_velocities_numerics[-1],
                        ideal = ideal_trajectory,
                        time_step = travel_time_laser_1_upper,
                        absolute_time = abs_time_upper,
                        additional_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        tungsten_positions = tungsten_positions,
                        test_mass = test_mass,
                    ),
                    travel_time_laser_1_upper,
                    internal_state_upper_path[index + 1],
                ]
            )
            upper_velocities_numerics.append(
                atom_velocity(
                    start_height = upper_heights_numerics[-1][0],
                    start_velocity = upper_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = travel_time_laser_1_upper,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )
            )

            lower_heights_numerics.append(
                [
                    atom_trajectory(
                        start_height = lower_heights_numerics[-1][0],
                        start_velocity = lower_velocities_numerics[-1],
                        ideal = ideal_trajectory,
                        time_step = travel_time_laser_1_lower,
                        absolute_time = abs_time_lower,
                        additional_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        tungsten_positions = tungsten_positions,
                        test_mass = test_mass,
                    ),
                    travel_time_laser_1_lower,
                    internal_state_lower_path[index + 1],
                ]
            )
            lower_velocities_numerics.append(
                atom_velocity(
                    start_height = lower_heights_numerics[-1][0],
                    start_velocity = lower_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = travel_time_laser_1_lower,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                )
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~ Next atom propagation ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        time_window = interval_length_num[index + 1] / time_discretization

        # Initial conditions for the next interferometry paths: The kicked atom is propagated through the first time window (as in the while-loop below), so that atom and absolute time counter stay in sync.
        if upper_relaunches is not None:
            relaunch_vel = Decimal(upper_relaunches[index + 1])
        else:
            relaunch_vel = Decimal(0)

        if internal_state_upper_path[index + 2] == "e":
            kick_velocity_upper = (
                next_interaction_velocity_upper
                + relaunch_vel
                + trans_momentum_upper_path[index + 1] * Decimal(hbar_over_m * k_effective / (1 + clock_mass_gain))
            )
        else:
            kick_velocity_upper = (
                next_interaction_velocity_upper
                + relaunch_vel
                + trans_momentum_upper_path[index + 1] * Decimal(hbar_over_m * k_effective)
            )

        upper_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = next_interaction_height_upper,
                    start_velocity = kick_velocity_upper,
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                ),
                time_window,
                internal_state_upper_path[index + 2],
            ]
        )
        upper_velocities_numerics.append(
            atom_velocity(
                start_height = upper_heights_numerics[-1][0],
                start_velocity = kick_velocity_upper,
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            )
        )

        abs_time_upper += time_window

        if lower_relaunches is not None:
            relaunch_vel = Decimal(lower_relaunches[index + 1])
        else:
            relaunch_vel = Decimal(0)

        if internal_state_lower_path[index + 2] == "e":
            kick_velocity_lower = (
                next_interaction_velocity_lower
                + relaunch_vel
                + trans_momentum_lower_path[index + 1] * Decimal(hbar_over_m * k_effective / (1 + clock_mass_gain))
            )
        else:
            kick_velocity_lower = (
                next_interaction_velocity_lower
                + relaunch_vel
                + trans_momentum_lower_path[index + 1] * Decimal(hbar_over_m * k_effective)
            )

        lower_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = next_interaction_height_lower,
                    start_velocity = kick_velocity_lower,
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                ),
                time_window,
                internal_state_lower_path[index + 2],
            ]
        )
        lower_velocities_numerics.append(
            atom_velocity(
                start_height = lower_heights_numerics[-1][0],
                start_velocity = kick_velocity_lower,
                time_step = time_window,
                ideal = ideal_trajectory,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            )
        )

        abs_time_lower += time_window

        # Step 2: While-loop propagation through full time windows, until only one more full time window would fit before the next laser pulse starts
        next_end_of_interval = Decimal(sum(interval_length_num[: (index + 2)]))

        while (
            abs_time_upper + time_window <= next_end_of_interval + absolute_time_offset
        ):
            upper_heights_numerics.append(
                [
                    atom_trajectory(
                        start_height = upper_heights_numerics[-1][0],
                        start_velocity = upper_velocities_numerics[-1],
                        time_step = time_window,
                        ideal = ideal_trajectory,
                        absolute_time = abs_time_upper,
                        additional_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        tungsten_positions = tungsten_positions,
                        test_mass = test_mass
                    ),
                    time_window,
                    internal_state_upper_path[index + 2],
                ]
            )
            upper_velocities_numerics.append(
                atom_velocity(
                    start_height = upper_heights_numerics[-1][0],
                    start_velocity = upper_velocities_numerics[-1],
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                )
            )
            abs_time_upper += time_window

        while (
            abs_time_lower + time_window <= next_end_of_interval + absolute_time_offset
        ):
            lower_heights_numerics.append(
                [
                    atom_trajectory(
                        start_height = lower_heights_numerics[-1][0],
                        start_velocity = lower_velocities_numerics[-1],
                        time_step = time_window,
                        ideal = ideal_trajectory,
                        absolute_time = abs_time_lower,
                        additional_acceleration = add_acceleration,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        tungsten_positions = tungsten_positions,
                        test_mass = test_mass
                    ),
                    time_window,
                    internal_state_lower_path[index + 2],
                ]
            )
            lower_velocities_numerics.append(
                atom_velocity(
                    start_height = lower_heights_numerics[-1][0],
                    start_velocity = lower_velocities_numerics[-1],
                    time_step = time_window,
                    ideal = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                )
            )
            abs_time_lower += time_window

        # Step 3: Final propagation step to reach the exact end of the interferometer path
        upper_time_till_next_laser_pulse = (next_end_of_interval + absolute_time_offset - abs_time_upper)
        lower_time_till_next_laser_pulse = (next_end_of_interval + absolute_time_offset - abs_time_lower)

        upper_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = upper_heights_numerics[-1][0],
                    start_velocity = upper_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = upper_time_till_next_laser_pulse,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass
                ),
                upper_time_till_next_laser_pulse,
                internal_state_upper_path[index + 2],
            ]
        )
        upper_velocities_numerics.append(
            atom_velocity(
                start_height = upper_heights_numerics[-1][0],
                start_velocity = upper_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = upper_time_till_next_laser_pulse,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass
            )
        )
        lower_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = lower_heights_numerics[-1][0],
                    start_velocity = lower_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = lower_time_till_next_laser_pulse,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                ),
                lower_time_till_next_laser_pulse,
                internal_state_lower_path[index + 2],
            ]
        )
        lower_velocities_numerics.append(
            atom_velocity(
                start_height = lower_heights_numerics[-1][0],
                start_velocity = lower_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = lower_time_till_next_laser_pulse,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
        )

        abs_time_upper += upper_time_till_next_laser_pulse
        abs_time_lower += lower_time_till_next_laser_pulse


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Last laser propagation: Upper/lower atomic path ~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """ For the second pulse we now need two relative time counters -- one for each atomic path. It keeps track of the time the photon(s) need to reach each path. """
    relative_time_counter_upper_path = Decimal(0)
    relative_time_counter_lower_path = Decimal(0)

    """ UPPER PATH CALCULATION """
    atom_height_upper_path_before = upper_heights_numerics[-1][0]
    atom_vel_upper_path_before = upper_velocities_numerics[-1]
    emission_time_upper = abs_time_upper  # Emission time of the pulse, see the first beam splitter.

    if include_FSL_contribution:
        """ Gravitational Wave Analysis:
        The laser heights are altered due to the presence of a gravitational wave. This is essentially the stretch of the baseline. """
        current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)
        current_upper_laser_height = upper_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

        virtual_upper_atomic_height = atom_height_upper_path_before * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

        """ For two-photon processes and single-photon processes with an upward momentum kick, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
        This duration is altered from the gravitational wave and is stored in this variable. """
        time_duration_photon_baseline = propagation_time_photon_full_baseline(
            height_upper = current_upper_laser_height,
            height_lower = current_lower_laser_height,
            time_counter = relative_time_counter_upper_path,
        )

        # if interaction_type[-1] == "Two_Photon" or (interaction_type[-1] == "Single_Photon" and trans_momentum_upper_path[-1] > 0):
        if interaction_type[-1] == "Two_Photon":
            """ To calculate the interaction times, one need to know the interaction process.
            If this is a two-photon process, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
            We therefore start by adding the above calculated time duration to both of the time counters. """
            relative_time_counter_upper_path += time_duration_photon_baseline
            abs_time_upper += time_duration_photon_baseline

            """ Then we can calculate the atom's position and velocity at the time when the photon reaches the lower mirror. """
            atom_height_upper_path = atom_trajectory(
                start_height = virtual_upper_atomic_height,
                start_velocity = atom_vel_upper_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_upper_path,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            atom_vel_upper_path = atom_velocity(
                start_height = atom_height_upper_path_before,
                start_velocity = atom_vel_upper_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_upper_path,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )

            """ We now calculate the new current laser heights, since the gravitational wave might have altered them during the photon travel time.
            This effect is miniscule, but for the sake of completeness we include it here. Note that we only need to recalculate the lower laser height. """
            current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_upper) / 2)

            """ Now the photon travels from the lower mirror up to the atom, which takes some additional time depending on the atom's position and velocity at this time. """
            time_delta = photon_atom_interaction_time(
                photon_start_height = current_lower_laser_height,
                light_direction = "up",
                target_height = atom_height_upper_path,
                target_velocity = atom_vel_upper_path,
                ideal_trajectory = ideal_trajectory,
                absolute_time = abs_time_upper,
                add_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
            )

            if time_delta.is_nan():
                return [None, None, None, None, None, None]

            relative_time_counter_upper_path += time_delta
            abs_time_upper += time_delta

        # elif interaction_type[-1] == "Single_Photon" and trans_momentum_upper_path[-1] < 0:
        elif interaction_type[-1] == "Single_Photon":
            """ In a single photon process, the photon travels directly from the upper laser down to the atom or from the lower laser up to the atom, depending on the light direction of this pulse.
            Note that the light direction is not the direction of the atomic kick: Stimulated emission kicks the atom against the light. """

            atom_height = virtual_upper_atomic_height
            atom_vel = atom_vel_upper_path_before

            if laser_direction[-1] == "down":
                """ The photon travels from the upper laser down to the atom, which takes some time depending on the atom's position and velocity at this time. """
                relative_time_counter_upper_path += photon_atom_interaction_time(
                    photon_start_height = current_upper_laser_height,
                    light_direction = "down",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if relative_time_counter_upper_path.is_nan():
                    return [None, None, None, None, None, None]

                abs_time_upper += photon_atom_interaction_time(
                    photon_start_height = current_upper_laser_height,
                    light_direction = "down",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if abs_time_upper.is_nan():
                    return [None, None, None, None, None, None]


            elif laser_direction[-1] == "up":
                """ The photon travels from the lower laser up to the atom, which takes some time depending on the atom's position and velocity at this time. """
                relative_time_counter_upper_path += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if relative_time_counter_upper_path.is_nan():
                    return [None, None, None, None, None, None]

                abs_time_upper += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_upper,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if abs_time_upper.is_nan():
                    return [None, None, None, None, None, None]


            else:
                raise ValueError("Single photon pulse without light direction. Set single_photon_laser_direction of the interferometer.")

        """ Now we know that the travel time of the photon(s) in the next beam splitter/mirror took relative_time_counter_upper_path seconds.
        From this we can calculate the atom's position and velocity at the time of the second interaction: """

        last_interaction_height_upper = atom_trajectory(
            start_height = atom_height_upper_path_before,
            start_velocity = atom_vel_upper_path_before,
            ideal = ideal_trajectory,
            time_step = relative_time_counter_upper_path,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )
        last_interaction_velocity_upper = atom_velocity(
            start_height = atom_height_upper_path_before,
            start_velocity = atom_vel_upper_path_before,
            ideal = ideal_trajectory,
            time_step = relative_time_counter_upper_path,
            absolute_time = abs_time_upper,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )

    else:
        last_interaction_height_upper = Decimal(atom_height_upper_path_before)
        last_interaction_velocity_upper = Decimal(atom_vel_upper_path_before)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Mirror Jitter Next Interaction Upper Path ~~~~~
    # ~~~ Laser Phase Noise Next Interaction Upper Path ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Doppler parameter
    if include_Doppler_effect:
        doppler_shift_upper = last_interaction_velocity_upper / c

    else:
        doppler_shift_upper = Decimal(0)

    if mirror_jitter_bool and interaction_type[-1] == "Two_Photon":
        jitter = Decimal(
            create_displacement_noise_at_time(
                random_phases = random_phases_mirror_jitter,
                time = abs_time_upper
            )
        )

        k_vector_Doppler_corrected = k_effective * (
            1 - Decimal(doppler_shift_upper / 2)
        ) + recoil_frequency * Decimal(doppler_shift_upper / c)
        kick_phase_jitter += (
            2 * trans_momentum_upper_path[-1] * k_vector_Doppler_corrected * jitter
        )

    if laser_phase_noise_bool:
        kick_phase_jitter += direction_sign(pulse = -1) * trans_momentum_upper_path[-1] * phase_noise(
            phase_noise_model_phases = random_phases_laser_noise,
            time = emission_time_upper
        )

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~ Setback everything to calculate the lower path independently now. ~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    """ LOWER PATH CALCULATION """
    atom_height_lower_path_before = lower_heights_numerics[-1][0]
    atom_vel_lower_path_before = lower_velocities_numerics[-1]
    emission_time_lower = abs_time_lower  # Emission time of the pulse, see the first beam splitter.

    if include_FSL_contribution:
        """ Now we reset the absolute time counter to the time right after the first interferometer segment, since we will now calculate the lower path interaction independently. """

        """ Gravitational Wave Analysis:
        The laser heights are altered due to the presence of a gravitational wave. This is essentially the stretch of the baseline. """
        current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)
        current_upper_laser_height = upper_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

        virtual_lower_atomic_height = atom_height_lower_path_before * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

        """ For two-photon processes and single-photon processes with an upward momentum kick, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
        This duration is altered from the gravitational wave and is stored in this variable. """
        time_duration_photon_baseline = propagation_time_photon_full_baseline(
            height_upper = current_upper_laser_height,
            height_lower = current_lower_laser_height,
            time_counter = relative_time_counter_lower_path,
        )

        # if interaction_type[-1] == "Two_Photon" or (interaction_type[-1] == "Single_Photon" and momentum_quanta_lower_path[-1] > 0):
        if interaction_type[-1] == "Two_Photon":
            """ To calculate the interaction times, one need to know the interaction process.
            If this is a two-photon process, the photon first travels the full baseline from the upper laser to the mirror and then back up to the atom.
            We therefore start by adding the above calculated time duration to both of the time counters. """
            relative_time_counter_lower_path += time_duration_photon_baseline
            abs_time_lower += time_duration_photon_baseline

            """ Then we can calculate the atom's position and velocity at the time when the photon reaches the lower mirror. """
            atom_height_lower_path = atom_trajectory(
                start_height = virtual_lower_atomic_height,
                start_velocity = atom_vel_lower_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_lower_path,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            atom_vel_lower_path = atom_velocity(
                start_height = atom_height_lower_path_before,
                start_velocity = atom_vel_lower_path_before,
                ideal = ideal_trajectory,
                time_step = relative_time_counter_lower_path,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )

            """ We now calculate the new current laser heights, since the gravitational wave might have altered them during the photon travel time.
            This effect is miniscule, but for the sake of completeness we include it here. Note that we only need to recalculate the lower laser height. """
            current_lower_laser_height = lower_laser_height * (Decimal(1) + GW_strain(gw = gw, time = abs_time_lower) / 2)

            """ Now the photon travels from the lower mirror up to the atom, which takes some additional time depending on the atom's position and velocity at this time. """
            relative_time_counter_lower_path += photon_atom_interaction_time(
                photon_start_height = current_lower_laser_height,
                light_direction = "up",
                target_height = atom_height_lower_path,
                target_velocity = atom_vel_lower_path,
                ideal_trajectory = ideal_trajectory,
                absolute_time = abs_time_lower,
                add_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
            )

            if relative_time_counter_lower_path.is_nan():
                return [None, None, None, None, None, None]

            abs_time_lower += photon_atom_interaction_time(
                photon_start_height = current_lower_laser_height,
                light_direction = "up",
                target_height = atom_height_lower_path,
                target_velocity = atom_vel_lower_path,
                ideal_trajectory = ideal_trajectory,
                absolute_time = abs_time_lower,
                add_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
            )

            if abs_time_lower.is_nan():
                return [None, None, None, None, None, None]


        # elif interaction_type[-1] == "Single_Photon" and momentum_quanta_upper_path[-1] < 0:
        elif interaction_type[-1] == "Single_Photon":
            """ In a single photon process, the photon travels directly from the upper laser down to the atom or from the lower laser up to the atom, depending on the light direction of this pulse.
            Note that the light direction is not the direction of the atomic kick: Stimulated emission kicks the atom against the light. """

            atom_height = virtual_lower_atomic_height
            atom_vel = atom_vel_lower_path_before

            if laser_direction[-1] == "down":
                """ The photon travels from the upper laser down to the atom, which takes some time depending on the atom's position and velocity at this time. """
                relative_time_counter_lower_path += photon_atom_interaction_time(
                    photon_start_height = current_upper_laser_height,
                    light_direction = "down",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )
                abs_time_lower += photon_atom_interaction_time(
                    photon_start_height = current_upper_laser_height,
                    light_direction = "down",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

            elif laser_direction[-1] == "up":
                """ The photon travels from the lower laser up to the atom, which takes some time depending on the atom's position and velocity at this time. """
                relative_time_counter_lower_path += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if relative_time_counter_lower_path.is_nan():
                    return [None, None, None, None, None, None]

                abs_time_lower += photon_atom_interaction_time(
                    photon_start_height = current_lower_laser_height,
                    light_direction = "up",
                    target_height = atom_height,
                    target_velocity = atom_vel,
                    ideal_trajectory = ideal_trajectory,
                    absolute_time = abs_time_lower,
                    add_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                )

                if abs_time_lower.is_nan():
                    return [None, None, None, None, None, None]

            else:
                raise ValueError("Single photon pulse without light direction. Set single_photon_laser_direction of the interferometer.")

        """ Now we know that the travel time of the photon(s) in the second beam splitter/mirror took relative_time_counter_lower_path seconds.
        From this we can calculate the atom's position and velocity at the time of the second interaction: """

        last_interaction_height_lower = atom_trajectory(
            start_height = atom_height_lower_path_before,
            start_velocity = atom_vel_lower_path_before,
            ideal = ideal_trajectory,
            time_step = relative_time_counter_lower_path,
            absolute_time = abs_time_lower,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )
        last_interaction_velocity_lower = atom_velocity(
            start_height = atom_height_lower_path_before,
            start_velocity = atom_vel_lower_path_before,
            ideal = ideal_trajectory,
            time_step = relative_time_counter_lower_path,
            absolute_time = abs_time_lower,
            additional_acceleration = add_acceleration,
            g = g,
            gravity_gradient = gravity_gradient,
            tungsten_positions = tungsten_positions,
            test_mass = test_mass,
        )

    else:
        last_interaction_height_lower = Decimal(atom_height_lower_path_before)
        last_interaction_velocity_lower = Decimal(atom_vel_lower_path_before)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Mirror Jitter Last Interaction Upper Path ~~~~~
    # ~~~ Laser Phase Noise Last Interaction Upper Path ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Doppler parameter
    if include_Doppler_effect:
        doppler_shift_lower = last_interaction_velocity_lower / c

    else:
        doppler_shift_lower = Decimal(0)

    if mirror_jitter_bool and interaction_type[-1] == "Two_Photon":
        jitter = Decimal(
            create_displacement_noise_at_time(
                random_phases = random_phases_mirror_jitter,
                time = abs_time_lower
            )
        )

        k_vector_Doppler_corrected = k_effective * (
            1 - Decimal(doppler_shift_lower / 2)
        ) + recoil_frequency * Decimal(doppler_shift_lower / c)

        kick_phase_jitter += (- 2 * trans_momentum_lower_path[-1] * k_vector_Doppler_corrected * jitter)

    if laser_phase_noise_bool:
        kick_phase_jitter += - direction_sign(pulse = -1) * trans_momentum_lower_path[-1] * phase_noise(
            phase_noise_model_phases = random_phases_laser_noise,
            time = emission_time_lower
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Imprinted Laser Phase ~~~~
    # ~~~ ("Spatial Kick Phase") ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """ The last laser pulse will alter the (initial) velocity of the atoms depending on the path and is stored in the following variables. """
    last_spatial_kick_phase_laser_contribution_Doppler_corrected_upper = (
        trans_momentum_upper_path[-1]
        * doppler_and_mass_corrected_kick_velocity(
            vel = last_interaction_velocity_upper,
            internal_state = internal_state_upper_path[-1]
        ).normalize()
    )
    last_spatial_kick_phase_laser_contribution_Doppler_corrected_lower = (
        trans_momentum_lower_path[-1]
        * doppler_and_mass_corrected_kick_velocity(
            vel = last_interaction_velocity_lower,
            internal_state = internal_state_lower_path[-1]
        ).normalize()
    )

    """ The last laser pulse will imprint a spatial phase on the atoms, which is given by Eq. (35) """
    upper_spatial_kick_phase_list.append(
        last_interaction_height_upper.normalize()
        * last_spatial_kick_phase_laser_contribution_Doppler_corrected_upper.normalize()
        / hbar_over_m
    )
    lower_spatial_kick_phase_list.append(
        last_interaction_height_lower.normalize()
        * last_spatial_kick_phase_laser_contribution_Doppler_corrected_lower.normalize()
        / hbar_over_m
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Photon Flight Times ~~~
    # ~~~ Temporal Kick Phase ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if include_FSL_contribution:
        """ Note that not all the following parameters are used for every interaction type. """
        travel_time_laser_1_upper = relative_time_counter_upper_path
        travel_time_laser_2_Doppler_upper = (current_upper_laser_height - last_interaction_height_upper) / c

        travel_time_laser_2_upper = (travel_time_laser_1_upper - travel_time_laser_2_Doppler_upper)

        travel_time_laser_1_lower = relative_time_counter_lower_path
        travel_time_laser_2_Doppler_lower = (current_upper_laser_height - last_interaction_height_lower) / c  # Note that the upper laser height is used here, since the photon from laser 2 always starts from the upper laser.

        travel_time_laser_2_lower = (travel_time_laser_1_lower - travel_time_laser_2_Doppler_lower)

        """ We store the temporal kick phase (i.e. the wt part of the phase imprint) for both paths in the following variables. """
        upper_imprinted_temporal_phase = Decimal(0)
        lower_imprinted_temporal_phase = Decimal(0)

        """ Denote the overall minus sign in the "temporal kick phase" here, since the phase of an EM field is given by ±kz - wt with w = |k|c. """
        if interaction_type[-1] == "Two_Photon":
            """ In this two-Photon case, both light fields need to be considered. """

            """ Comment on Doppler shift convention: Laser one travels up (after reflection) and laser two travels down.
            The signs for the Doppler effect are therefore positive for laser one and negative for laser two.
            Because laser one has a positive k vector (traveling upwards), atoms with positive velocity are "moving away" from the light field -- hence the atoms see a lower frequency of the light field one.
            The light field is therefore detuned in the opposite direction, i.e. it is tuned with a positive Doppler shift. See Eq. (33) in the manuscript.
            For the second laser it is the other way around. And with an additional global minus sign, because the photon is emitted into the light field, the phase is therefore substracted from the atom.
            If the atom is kicked downwards, this is taken into account by the trans_momentum variables -- emission and absorption then switch roles. """

            # ~~~~~~~~~~~~~~~~~~
            # ~~~ Upper Path ~~~
            # ~~~~~~~~~~~~~~~~~~

            # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
            upper_imprinted_temporal_phase += (
                - trans_momentum_upper_path[-1] * travel_time_laser_1_upper * k_vector_1 * (1 + doppler_shift_upper) * c
            )

            # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
            upper_imprinted_temporal_phase += (
                trans_momentum_upper_path[-1] * travel_time_laser_2_upper * k_vector_2 * c
            )

            upper_imprinted_temporal_phase += (
                trans_momentum_upper_path[-1] * travel_time_laser_2_Doppler_upper * k_vector_2 * (1 - doppler_shift_upper) * c
            )

            # ~~~~~~~~~~~~~~~~~~
            # ~~~ Lower Path ~~~
            # ~~~~~~~~~~~~~~~~~~

            # Laser 1: Travels down and then up to the atom. For its whole travel time it is Doppler detuned.
            lower_imprinted_temporal_phase += (
                - trans_momentum_lower_path[-1] * travel_time_laser_1_lower * k_vector_1 * (1 + doppler_shift_lower) * c
            )

            # Laser 2: Two contributions, one for the travel time down (no Doppler detuning) and one for the travel time up (with Doppler detuning).
            lower_imprinted_temporal_phase += (
                trans_momentum_lower_path[-1] * travel_time_laser_2_lower * k_vector_2 * c
            )

            lower_imprinted_temporal_phase += (
                trans_momentum_lower_path[-1] * travel_time_laser_2_Doppler_lower * k_vector_2 * (1 - doppler_shift_lower) * c
            )

        elif interaction_type[-1] == "Single_Photon":
            """ Single photon pulse: Each arm picks up the temporal part of the laser phase of the light travelling in the direction laser_direction[-1], see single_photon_temporal_kick_phase.
            Absorption (g -> e) and stimulated emission (e -> g) differ in the sign of the momentum transfer relative to the light direction. """
            upper_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                arm = "upper",
                pulse = -1,
                interaction_height = last_interaction_height_upper,
                doppler_shift = doppler_shift_upper,
            )
            lower_imprinted_temporal_phase += single_photon_temporal_kick_phase(
                arm = "lower",
                pulse = -1,
                interaction_height = last_interaction_height_lower,
                doppler_shift = doppler_shift_lower,
            )

        upper_temporal_kick_phase_list.append(
            upper_imprinted_temporal_phase
        )

        lower_temporal_kick_phase_list.append(
            lower_imprinted_temporal_phase
        )

        upper_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = upper_heights_numerics[-1][0],
                    start_velocity = upper_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = travel_time_laser_1_upper,
                    absolute_time = abs_time_upper,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                ),
                travel_time_laser_1_upper,
                internal_state_upper_path[index + 2],
            ]
        )
        upper_velocities_numerics.append(
            atom_velocity(
                start_height = upper_heights_numerics[-1][0],
                start_velocity = upper_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = travel_time_laser_1_upper,
                absolute_time = abs_time_upper,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
        )
        lower_heights_numerics.append(
            [
                atom_trajectory(
                    start_height = lower_heights_numerics[-1][0],
                    start_velocity = lower_velocities_numerics[-1],
                    ideal = ideal_trajectory,
                    time_step = travel_time_laser_1_lower,
                    absolute_time = abs_time_lower,
                    additional_acceleration = add_acceleration,
                    g = g,
                    gravity_gradient = gravity_gradient,
                    tungsten_positions = tungsten_positions,
                    test_mass = test_mass,
                ),
                travel_time_laser_1_lower,
                internal_state_lower_path[index + 2],
            ]
        )
        lower_velocities_numerics.append(
            atom_velocity(
                start_height = lower_heights_numerics[-1][0],
                start_velocity = lower_velocities_numerics[-1],
                ideal = ideal_trajectory,
                time_step = travel_time_laser_1_lower,
                absolute_time = abs_time_lower,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
        )

        # The absolute time counters already contain the photon travel time (see the photon timing of the last pulse above), as for the other pulses.
        # Adding it a second time would let both arms end at different times for detection_time > 0, which the rest energy m c^2 of the internal states amplifies by m c^2 / hbar.

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Last atom propagation ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Length of the last interval between two pulses (time_intervals has one entry per interval between two consecutive pulses).
    # It sets the time windows of the propagation until the detection; the stored time windows and the trajectory steps have to use the same length.
    last_interval_length = interval_length_num[-1]
    time_window = last_interval_length / time_discretization

    # Velocities directly after the last pulse. The separation phase needs them for every detection time, also if there is no further propagation.
    if internal_state_upper_path[-2] == "e":
        output_velocity_upper = last_interaction_velocity_upper + trans_momentum_upper_path[-1] * Decimal(hbar_over_m * k_effective / (1 + clock_mass_gain))
    else:
        output_velocity_upper = last_interaction_velocity_upper + trans_momentum_upper_path[-1] * Decimal(hbar_over_m * k_effective)

    if internal_state_lower_path[-2] == "e":
        output_velocity_lower = last_interaction_velocity_lower + trans_momentum_lower_path[-1] * Decimal(hbar_over_m * k_effective / (1 + clock_mass_gain))
    else:
        output_velocity_lower = last_interaction_velocity_lower + trans_momentum_lower_path[-1] * Decimal(hbar_over_m * k_effective)

    # State directly after the last pulse: No time passes (time step 0), but the velocities include the momentum transfer of the last pulse.
    upper_heights_numerics.append([last_interaction_height_upper, Decimal(0), internal_state_upper_path[-1]])
    upper_velocities_numerics.append(output_velocity_upper)
    lower_heights_numerics.append([last_interaction_height_lower, Decimal(0), internal_state_lower_path[-1]])
    lower_velocities_numerics.append(output_velocity_lower)

    # Detection: Both arms end at the same time, detection_time after the interaction of the later arm with the last pulse. The earlier arm catches up with the later one,
    # i.e. the arms are only propagated forwards. The propagation uses time windows of the length of the last interval and one shorter step for the remainder.
    upper_duration = sum((entry[1] for entry in upper_heights_numerics), Decimal(0))
    lower_duration = sum((entry[1] for entry in lower_heights_numerics), Decimal(0))
    end_of_propagation = max(upper_duration, lower_duration) + detection_time

    def propagate_until_detection(
        heights: list[list],  # Height list of the arm: [height, time window, internal state] per step.
        velocities: list[Decimal],  # Velocity list of the arm.
        absolute_time: Decimal,  # Absolute time at the start of the propagation.
        remaining_time: Decimal,  # Time until the detection.
        internal_state: str,  # Internal state after the last pulse.
    ) -> Decimal:

        """ Appends the propagation of one arm until the detection to its height and velocity lists and returns the absolute time at the detection. """
        while remaining_time > 0:
            step = min(time_window, remaining_time) if time_window > 0 else remaining_time
            height = atom_trajectory(
                start_height = heights[-1][0],
                start_velocity = velocities[-1],
                ideal = ideal_trajectory,
                time_step = step,
                absolute_time = absolute_time,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            velocity = atom_velocity(
                start_height = height,
                start_velocity = velocities[-1],
                ideal = ideal_trajectory,
                time_step = step,
                absolute_time = absolute_time,
                additional_acceleration = add_acceleration,
                g = g,
                gravity_gradient = gravity_gradient,
                tungsten_positions = tungsten_positions,
                test_mass = test_mass,
            )
            heights.append([height, step, internal_state])
            velocities.append(velocity)
            absolute_time += step
            remaining_time -= step

        return absolute_time

    abs_time_upper = propagate_until_detection(
        heights = upper_heights_numerics,
        velocities = upper_velocities_numerics,
        absolute_time = abs_time_upper,
        remaining_time = end_of_propagation - upper_duration,
        internal_state = internal_state_upper_path[-1],
    )
    abs_time_lower = propagate_until_detection(
        heights = lower_heights_numerics,
        velocities = lower_velocities_numerics,
        absolute_time = abs_time_lower,
        remaining_time = end_of_propagation - lower_duration,
        internal_state = internal_state_lower_path[-1],
    )

    # ~~~~~~~~~~~~~~~~~~
    # ~~~ Kick phase ~~~
    # ~~~~~~~~~~~~~~~~~~

    # Final temporal kick phase
    temporal_kick_phase = sum(upper_temporal_kick_phase_list) - sum(lower_temporal_kick_phase_list)

    # Final spatial kick phase
    spatial_kick_phase = sum(upper_spatial_kick_phase_list) - sum(lower_spatial_kick_phase_list)

    full_kick_phase = spatial_kick_phase + temporal_kick_phase + kick_phase_jitter

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Separation phase ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    # Calculation of the separation phase
    aver_output_velocity = (upper_velocities_numerics[-1] + lower_velocities_numerics[-1]) / 2
    separation = lower_heights_numerics[-1][0] - upper_heights_numerics[-1][0]

    separation_phase = separation * aver_output_velocity / hbar_over_m

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Propagation phase ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # Calculation of the propagation phase
    upper_prop_phase_numerics = Decimal(0)
    time_counter = absolute_time_offset

    for time_index in range(len(upper_heights_numerics)):
        # Since the time index runs through all (2 * time_discretization + 1)-steps, we need to divide by the time discretization to get the correct interval number, i.e. 1, 2, or 3.
        # Note that the internal state before the interferometer starts, i.e. object internal_state_upper_path[0], is irrelevant for the propagation phase and therefore not called.

        if upper_heights_numerics[time_index][2] == "e":

            # If the internal state is the excited state, we need to include the clock mass gain.
            clock_term = clock_mass_gain

        else:
            clock_term = Decimal(0)

        # The propagation phase, which is usually given as the integral over the Lagrangian, via infinitesimal sums over time steps.
        # The first part is the kinetic energy minus and the second part is the gravitational potential energy, which is imported from the gravitational_model.py file.
        # With internal states, the Lagrangian L = - m c^2 + m v^2 / 2 - m phi also contains the rest energy. The mass m (1 + clock_term) of the excited state gives the phase - clock_frequency * (time in the excited state).
        time_counter += upper_heights_numerics[time_index][1]

        if include_internal_state_changes:
            if only_tungsten:
                upper_prop_phase_numerics += (
                    - c ** 2 - model_phi(
                        h = upper_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(upper_heights_numerics[time_index][1] * (1 + clock_term))

            else:
                upper_prop_phase_numerics += (
                    - c ** 2 + (upper_velocities_numerics[time_index] ** 2) / 2
                    - model_phi(
                        h = upper_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(upper_heights_numerics[time_index][1] * (1 + clock_term))

        else:
            if only_tungsten:
                upper_prop_phase_numerics += - model_phi(
                        h = upper_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    ) * Decimal(upper_heights_numerics[time_index][1] * (1 + clock_term))
            else:
                upper_prop_phase_numerics += (
                    (upper_velocities_numerics[time_index] ** 2) / 2
                    - model_phi(
                        h = upper_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(upper_heights_numerics[time_index][1] * (1 + clock_term))

    # We perform the same steps for the lower interferometry path.
    lower_prop_phase_numerics = Decimal(0)
    time_counter = absolute_time_offset

    for time_index in range(len(lower_heights_numerics)):
        if lower_heights_numerics[time_index][2] == "e":

            # If the internal state is the excited state, we need to include the clock mass gain.
            clock_term = clock_mass_gain

        else:
            clock_term = Decimal(0)

        time_counter += lower_heights_numerics[time_index][1]

        if include_internal_state_changes:
            if only_tungsten:
                lower_prop_phase_numerics += (
                    - c ** 2 - model_phi(
                        h = lower_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(lower_heights_numerics[time_index][1] * (1 + clock_term))

            else:
                lower_prop_phase_numerics += (
                    - c ** 2 + (lower_velocities_numerics[time_index] ** 2) / 2
                    - model_phi(
                        h = lower_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(lower_heights_numerics[time_index][1] * (1 + clock_term))
        else:
            if only_tungsten:
                lower_prop_phase_numerics += - model_phi(
                        h = lower_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    ) * Decimal(lower_heights_numerics[time_index][1] * (1 + clock_term))
            else:
                lower_prop_phase_numerics += (
                    (lower_velocities_numerics[time_index] ** 2) / 2
                    - model_phi(
                        h = lower_heights_numerics[time_index][0],
                        ideal = ideal_potential,
                        absolute_time = time_counter,
                        additional_global_acceleration = add_acceleration,
                        tungsten_positions = tungsten_positions,
                        g = g,
                        gravity_gradient = gravity_gradient,
                        test_mass = test_mass,
                        only_tungsten = only_tungsten,
                        GGN_noise = GGN_noise,
                        z_offset = z_offset
                    )) * Decimal(lower_heights_numerics[time_index][1] * (1 + clock_term))

    # The propagation phase is then just the difference between the upper and lower path. We still need to divide by hbar/m to get the correct units, since this prefactor was left out of the Lagrangian.
    propagation_phase = (upper_prop_phase_numerics - lower_prop_phase_numerics) / hbar_over_m

    if print_trajectory:
        import matplotlib.pyplot as mpl

        upper_plot_times = [0.0]
        upper_time = Decimal(0)
        for time_index in range(len(upper_heights_numerics)):
            upper_time += upper_heights_numerics[time_index][1]
            upper_plot_times.append(float(upper_time))

        lower_plot_times = [0.0]
        lower_time = Decimal(0)
        for time_index in range(len(lower_heights_numerics)):
            lower_time += lower_heights_numerics[time_index][1]
            lower_plot_times.append(float(lower_time))

        mpl.figure()
        mpl.xlabel("$t$ in [s]", size = 15)
        mpl.ylabel("$z$ in [m]", size = 15)

        # for index in range(len(interval_length_num)):
        #     mpl.axvline(x = 0.0, color = "b", label = "axvline - full height")
        #     mpl.axvline(x = float(sum(interval_length_num[:(index + 1)])), color = "b", label = "axvline - full height")

        mpl.step(
            upper_plot_times,
            [float(initial_height)] + [float(item[0]) for item in upper_heights_numerics],
            "-",
            color = "black",
            linewidth = 1,
        )
        mpl.step(
            lower_plot_times,
            [float(initial_height)] + [float(item[0]) for item in lower_heights_numerics],
            "-",
            color = "red",
            linewidth = 1,
        )

        mpl.show()

    #print("Prop, Sep, Kick: ", propagation_phase, separation_phase, full_kick_phase)
    return [
        propagation_phase,
        separation_phase,
        full_kick_phase,
        upper_heights_numerics,
        lower_heights_numerics,
    ]


# Outputs the phase shift of a single interferometer and returns a specific phase shift origin, if requested
def num_phase_with_origin(
    interferometer: Interferometer,
    init_height: Decimal | float | int,
    init_velocity: Decimal | float | int,
    ideal_potential: str = "Homogeneous",
    ideal_trajectory: str = "Homogeneous",
    phase_output: str = "All",
    detection_time: Decimal = Decimal(0),
    absolute_time_offset: Decimal = Decimal(0),
    random_phases_laser_noise: list | None = None,
    random_phases_mirror_jitter: list | None = None,
    add_gravity: Decimal | float | int | None = None,
    gw: GravWave = zeroGW,
    g: Decimal | float | int = DEFAULT_G,
    gravity_gradient: Decimal | float | int = DEFAULT_GRAVITY_GRADIENT,
    include_FSL_contribution: bool = True,
    include_Doppler_effect: bool = True,
    include_internal_state_changes: bool = False,
    test_mass: TestMass | None = None,
    tungsten_positions: list | float | None = None,
    only_tungsten: bool = False,
    print_trajectory: bool = False,
    GGN_noise: GGN_Noise_Model | None = None,
    z_offset: float = 10.0,
    time_discretization: int = 1000,
) -> Decimal | None:

    phases = numerical_phases(
        interferometer = interferometer,
        initial_height = init_height,
        initial_velocity = init_velocity,
        ideal_potential = ideal_potential,
        ideal_trajectory = ideal_trajectory,
        detection_time = detection_time,
        absolute_time_offset = absolute_time_offset,
        random_phases_laser_noise = random_phases_laser_noise,
        random_phases_mirror_jitter = random_phases_mirror_jitter,
        add_gravity = add_gravity,
        gw = gw,
        g = g,
        gravity_gradient = gravity_gradient,
        include_FSL_contribution = include_FSL_contribution,
        include_Doppler_effect = include_Doppler_effect,
        include_internal_state_changes = include_internal_state_changes,
        tungsten_positions = tungsten_positions,
        test_mass = test_mass,
        only_tungsten = only_tungsten,
        print_trajectory = print_trajectory,
        GGN_noise = GGN_noise,
        z_offset = z_offset,
        time_discretization = time_discretization
    )

    if phases[0] is None:
        return None

    if phase_output == "Propagation":
        return phases[0]

    elif phase_output == "Separation":
        return phases[1]

    elif phase_output == "Kick":
        return phases[2]

    elif phase_output == "Upper":
        return phases[3]

    elif phase_output == "Lower":
        return phases[4]

    else:
        # Write output_type == "All"
        return phases[0] + phases[1] + phases[2]


# Both interferometers of a differential measurement are driven by the same lasers, i.e. they need the same laser geometry.
def check_same_laser_geometry(
    interferometer_1: Interferometer,
    interferometer_2: Interferometer
) -> None:

    if Decimal(interferometer_1.lower_laser_height) != Decimal(interferometer_2.lower_laser_height) or Decimal(interferometer_1.baseline) != Decimal(interferometer_2.baseline):
        raise ValueError(
            f"The interferometers {interferometer_1.name} and {interferometer_2.name} share the lasers and need the same lower_laser_height and baseline "
            f"(given: {interferometer_1.lower_laser_height} m and {interferometer_1.baseline} m, {interferometer_2.lower_laser_height} m and {interferometer_2.baseline} m)."
        )


# Output the phase shift of a differential measurement between two interferometers
def numerical_differential_interferometer(
    interferometer_1: Interferometer,
    interferometer_2: Interferometer,
    init_height: Decimal,
    init_velocity: Decimal,
    ideal_potential: str = "Homogeneous",
    ideal_trajectory: str = "Homogeneous",
    phase_output: str = "All",
    detection_time: Decimal = Decimal(0),
    absolute_time_offset: Decimal = Decimal(0),
    random_phases_laser_noise: list | None = None,
    random_phases_mirror_jitter: list | None = None,
    add_gravity: Decimal | float | int | None = None,
    gw: GravWave = zeroGW,
    g: Decimal | float | int = DEFAULT_G,
    gravity_gradient: Decimal | float | int = DEFAULT_GRAVITY_GRADIENT,
    include_FSL_contribution: bool = True,
    include_Doppler_effect: bool = True,
    include_internal_state_changes: bool = False,
    test_mass: TestMass | None = None,
    tungsten_positions: list | float | None = None,
    only_tungsten: bool = False,
    GGN_noise: GGN_Noise_Model | None = None,
    z_offset: float = 10.0,
    print_trajectory: bool = False
) -> Decimal | None:

    check_same_laser_geometry(interferometer_1 = interferometer_1, interferometer_2 = interferometer_2)

    if1 = num_phase_with_origin(
        interferometer = interferometer_1,
        ideal_potential = ideal_potential,
        ideal_trajectory = ideal_trajectory,
        init_height = init_height,
        init_velocity = init_velocity,
        phase_output = phase_output,
        detection_time = detection_time,
        absolute_time_offset = absolute_time_offset,
        random_phases_laser_noise = random_phases_laser_noise,
        random_phases_mirror_jitter = random_phases_mirror_jitter,
        add_gravity = add_gravity,
        gw = gw,
        g = g,
        gravity_gradient = gravity_gradient,
        include_FSL_contribution = include_FSL_contribution,
        include_Doppler_effect = include_Doppler_effect,
        include_internal_state_changes = include_internal_state_changes,
        tungsten_positions = tungsten_positions,
        test_mass = test_mass,
        only_tungsten = only_tungsten,
        GGN_noise = GGN_noise,
        z_offset = z_offset,
        print_trajectory = print_trajectory
    )

    if2 = num_phase_with_origin(
        interferometer = interferometer_2,
        ideal_potential = ideal_potential,
        ideal_trajectory = ideal_trajectory,
        init_height = init_height,
        init_velocity = init_velocity,
        phase_output = phase_output,
        detection_time = detection_time,
        absolute_time_offset = absolute_time_offset,
        random_phases_laser_noise = random_phases_laser_noise,
        random_phases_mirror_jitter = random_phases_mirror_jitter,
        add_gravity = add_gravity,
        gw = gw,
        g = g,
        gravity_gradient = gravity_gradient,
        include_FSL_contribution = include_FSL_contribution,
        include_Doppler_effect = include_Doppler_effect,
        include_internal_state_changes = include_internal_state_changes,
        tungsten_positions = tungsten_positions,
        test_mass = test_mass,
        only_tungsten = only_tungsten,
        GGN_noise = GGN_noise,
        z_offset = z_offset,
        print_trajectory = print_trajectory
    )

    if if1 is None or if2 is None:
        return None
    else:
        return if1 - if2


def numerical_same_interferometer_height_difference(
    interferometer: Interferometer,
    height_delta: Decimal,
    init_height: Decimal,
    init_velocity: Decimal,
    ideal_potential: str = "Homogeneous",
    ideal_trajectory: str = "Homogeneous",
    phase_output: str = "All",
    detection_time: Decimal = Decimal(0),
    absolute_time_offset: Decimal = Decimal(0),
    random_phases_laser_noise: list | None = None,
    random_phases_mirror_jitter: list | None = None,
    add_gravity: Decimal | float | int | None = None,
    gw: GravWave = zeroGW,
    g: Decimal | float | int = DEFAULT_G,
    gravity_gradient: Decimal | float | int = DEFAULT_GRAVITY_GRADIENT,
    include_FSL_contribution: bool = True,
    include_Doppler_effect: bool = True,
    include_internal_state_changes: bool = False,
    test_mass: TestMass | None = None,
    tungsten_positions: list | float | None = None,
    only_tungsten: bool = False,
    GGN_noise: GGN_Noise_Model | None = None,
    print_trajectory: bool = False,
) -> Decimal | None:

    if1 = num_phase_with_origin(
        interferometer = interferometer,
        ideal_potential = ideal_potential,
        ideal_trajectory = ideal_trajectory,
        init_height = init_height,
        init_velocity = init_velocity,
        phase_output = phase_output,
        detection_time = detection_time,
        absolute_time_offset = absolute_time_offset,
        random_phases_laser_noise = random_phases_laser_noise,
        random_phases_mirror_jitter = random_phases_mirror_jitter,
        add_gravity = add_gravity,
        gw = gw,
        g = g,
        gravity_gradient = gravity_gradient,
        include_FSL_contribution = include_FSL_contribution,
        include_Doppler_effect = include_Doppler_effect,
        include_internal_state_changes = include_internal_state_changes,
        tungsten_positions = tungsten_positions,
        test_mass = test_mass,
        only_tungsten = only_tungsten,
        GGN_noise = GGN_noise,
        print_trajectory = print_trajectory
    )

    if2 = num_phase_with_origin(
        interferometer = interferometer,
        ideal_potential = ideal_potential,
        ideal_trajectory = ideal_trajectory,
        init_height = init_height + height_delta,
        init_velocity = init_velocity,
        phase_output = phase_output,
        detection_time = detection_time,
        absolute_time_offset = absolute_time_offset,
        random_phases_laser_noise = random_phases_laser_noise,
        random_phases_mirror_jitter = random_phases_mirror_jitter,
        add_gravity = add_gravity,
        gw = gw,
        g = g,
        gravity_gradient = gravity_gradient,
        include_FSL_contribution = include_FSL_contribution,
        include_Doppler_effect = include_Doppler_effect,
        include_internal_state_changes = include_internal_state_changes,
        tungsten_positions = tungsten_positions,
        test_mass = test_mass,
        only_tungsten = only_tungsten,
        GGN_noise = GGN_noise,
        print_trajectory = print_trajectory
    )

    if if1 is None or if2 is None:
        return None
    else:
        return if1 - if2


# END OF FILE
