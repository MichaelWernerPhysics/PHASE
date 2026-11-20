from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from astropy import coordinates
from astropy.time import Time
from scipy.constants import day, hour

from phase.decimal_additional_functions import pi, sin

# ~~~~~~~~~~~~~~~
# ~~~ Classes ~~~
# ~~~~~~~~~~~~~~~


# Interferometer class definition
class Interferometer:
    def __init__(
            self,
            # Name of the Interferometer
            name: str,
            # List of time intervals between consecutive pulses: P - 1 entries for P pulses. The propagation after the last pulse is the detection_time of numerical_phases.
            time_intervals: Sequence[int | float | Decimal],
            # List of momentum states within the upper path (for each segment + initial + final)
            upper_momentum: list[int],
            # List of momentum states within the lower path (for each segment + initial + final)
            lower_momentum: list[int],
            # List of internal states within the upper path (for each segment + initial + final)
            upper_internal_state: list[str],
            # List of internal states within the lower path (for each segment + initial + final)
            lower_internal_state: list[str],
            # List of interaction types with each laser pulse. Either "Single_Photon" or "Two_Photon".
            interaction_type: list[str],
            # Dimensionless detunings: pulse i scales the momentum transfer in both arms by (1 + detunings[i]).
            detunings: Sequence[int | float | Decimal],
            # k-vector in 1/m, no default. Single photon pulses: the k-vector of the photon. Two-photon pulses: the k-vectors of both beams, [k_1, k_2] with k_eff = k_1 + k_2,
            # e.g. [k / 2, k / 2] with k = 4 pi / 780 nm for the Raman transition of Rubidium. For single photon pulses with the finite speed of light, k c has to equal clock_frequency.
            k_vector: int | float | Decimal | list,
            # Default value is for Rubidium-87, which is commonly used in atom interferometry.
            nucleon_number: int = 87,
            # List of strings or None to indicate which noise sources to include. Noise list convention includes "Mirror Jitter" and "Laser Phase" (fluctuations of the gravitational field are included via GGN_noise of numerical_phases). So an example noise list would be ["Mirror Jitter"] if we want to include the mirror jitter, but not laser phase noise.
            noise: list[str] | None = None,
            # Default value for Rubidium. Angular frequency in rad/s.
            clock_frequency: int | float | Decimal = Decimal(2 * np.pi * 6.8 * 10 ** 9),
            # Height in m of the lower laser (emits the light travelling up) or of the retro-reflection mirror (two-photon pulses).
            lower_laser_height: int | float | Decimal = -1,
            # Distance in m between the lower and the upper laser: the upper laser (emits the light travelling down) sits at lower_laser_height + baseline.
            baseline: int | float | Decimal = 12,
            # List of upper and lower relaunches in m/s
            upper_relaunches: list[int | float | Decimal] | None = None,
            lower_relaunches: list[int | float | Decimal] | None = None,
            # Propagation direction of the light for each laser pulse: "Up" (emitted by the lower laser) or "Down" (emitted by the upper laser) for "Single_Photon" pulses, None for "Two_Photon" pulses.
            # This is the direction of the light, not of the atomic kick: Absorption kicks the atom along the light, stimulated emission against it.
            # Only needed with the finite speed of light. If None, every "Single_Photon" pulse is left without a direction.
            single_photon_laser_direction: Sequence[str | None] | None = None
        ) -> None:

        self.name = name
        self.time_intervals = time_intervals
        self.upper_momentum = upper_momentum
        self.lower_momentum = lower_momentum
        self.upper_internal_state = upper_internal_state
        self.lower_internal_state = lower_internal_state
        self.interaction_type = interaction_type
        self.detunings = detunings
        self.nucleon_number = nucleon_number
        self.noise = noise
        self.clock_frequency = clock_frequency
        self.k_vector = k_vector
        self.lower_laser_height = lower_laser_height
        self.baseline = baseline
        self.upper_relaunches = upper_relaunches
        self.lower_relaunches = lower_relaunches
        self.single_photon_laser_direction = single_photon_laser_direction

        # Validate input data: For P laser pulses, the momentum and internal state lists have P + 1 entries (before, between and after the pulses),
        # interaction type and detuning have P entries (one per pulse) and time_intervals has P - 1 entries (one per interval between two consecutive pulses).
        # The propagation after the last pulse is given by the detection_time of numerical_phases.
        number_of_pulses = len(interaction_type)
        path_lists: dict[str, Sequence[Any]] = {"upper_momentum": upper_momentum, "lower_momentum": lower_momentum, "upper_internal_state": upper_internal_state, "lower_internal_state": lower_internal_state}

        wrong_lengths = [f"{key} has {len(value)}" for key, value in path_lists.items() if len(value) != number_of_pulses + 1]
        if len(detunings) != number_of_pulses:
            wrong_lengths.append(f"detunings has {len(detunings)}")
        if wrong_lengths:
            raise ValueError(
                f"Interferometer {name}: With {number_of_pulses} pulses (length of interaction_type), the momentum and internal state lists need {number_of_pulses + 1} entries "
                f"and detunings {number_of_pulses}, but " + ", ".join(wrong_lengths) + "."
            )

        if len(time_intervals) != number_of_pulses - 1:
            raise ValueError(
                f"Interferometer {name}: With {number_of_pulses} pulses, time_intervals needs {number_of_pulses - 1} entries (one per interval between two consecutive pulses), "
                f"but has {len(time_intervals)}. The propagation after the last pulse is given by the detection_time of numerical_phases."
            )

        if isinstance(k_vector, list) and len(k_vector) != 2:
            raise ValueError(f"Interferometer {name}: A list of k-vectors needs the two beams of a two-photon pulse, k_vector = [k_1, k_2] (given: {len(k_vector)} entries).")
        if "Two_Photon" in interaction_type and not isinstance(k_vector, list):
            raise ValueError(f"Interferometer {name}: Two-photon pulses need the k-vectors of both beams, k_vector = [k_1, k_2] with k_eff = k_1 + k_2 (given: {k_vector}).")

        if Decimal(baseline) <= 0:
            raise ValueError(f"Interferometer {name}: The baseline is the distance between the lower and the upper laser and has to be positive (given: {baseline}).")

        if single_photon_laser_direction is not None:
            self._validate_single_photon_laser_direction()

    @property
    def upper_laser_height(self) -> Decimal:
        """ Height in m of the upper laser: one baseline above the lower laser (or retro-reflection mirror). """
        return Decimal(self.lower_laser_height) + Decimal(self.baseline)

    def _validate_single_photon_laser_direction(self) -> None:

        """ Checks the light direction of each pulse against the momentum transfer and the internal state change.
        Light travelling in the direction s (+1 = up, -1 = down) kicks the atom by +s ħk when it absorbs a photon (g -> e) and by -s ħk under stimulated emission (e -> g).
        Pulses in which an arm changes its internal state and receives momentum therefore fix the direction s = sign(Δp) for g -> e and s = -sign(Δp) for e -> g.
        Arms without internal state change carry no information about the direction and are not checked. """

        directions = self.single_photon_laser_direction
        assert directions is not None

        if len(directions) != len(self.interaction_type):
            raise ValueError(
                f"Interferometer {self.name}: single_photon_laser_direction needs one entry per laser pulse ({len(self.interaction_type)}), but has {len(directions)}."
            )

        for pulse, (interaction, direction) in enumerate(zip(self.interaction_type, directions, strict = True)):
            if interaction != "Single_Photon":
                if direction is not None:
                    raise ValueError(
                        f"Interferometer {self.name}: Pulse {pulse} is a {interaction} pulse, its single_photon_laser_direction has to be None (given: {direction!r})."
                    )
                continue

            if not isinstance(direction, str) or direction.lower() not in ("up", "down"):
                raise ValueError(
                    f"Interferometer {self.name}: Single photon pulse {pulse} needs the light direction \"Up\" or \"Down\" (given: {direction!r})."
                )

            for arm, momentum, states in (("upper", self.upper_momentum, self.upper_internal_state), ("lower", self.lower_momentum, self.lower_internal_state)):
                momentum_transfer = momentum[pulse + 1] - momentum[pulse]
                transition = (states[pulse], states[pulse + 1])

                if momentum_transfer == 0 or transition not in (("g", "e"), ("e", "g")):
                    continue

                absorption = transition == ("g", "e")
                required_direction = "up" if (momentum_transfer > 0) == absorption else "down"

                if direction.lower() != required_direction:
                    raise ValueError(
                        f"Interferometer {self.name}: In pulse {pulse}, the {arm} arm changes from {transition[0]} to {transition[1]} with a momentum transfer of {momentum_transfer} ħk. "
                        f"{'Absorption' if absorption else 'Stimulated emission'} kicks the atom {'along' if absorption else 'against'} the light, i.e. the light has to travel {required_direction}, "
                        f"but single_photon_laser_direction[{pulse}] = {direction!r}."
                    )


class location:
    def __init__(
            self,
            starttime: int | float | datetime,
            location_lon: int | float,
            location_lat: int | float,
            name: str = "Test",
            length: int | float = 14 * day,
            resolution: int | float = hour
        ) -> None:

        self.starttime = Time(starttime, format = 'datetime', scale = 'utc')
        self.location_lon = location_lon
        self.location_lat = location_lat
        self.location = coordinates.EarthLocation.from_geodetic(location_lon,location_lat)
        position = []
        distance = []

        self.length = length
        self.resolution = resolution # (seconds)

        self.name = name

        for i in range(int(self.length/self.resolution)):

            altazframe = coordinates.AltAz(obstime = self.starttime+timedelta(seconds = self.resolution*i), location = self.location)
            self.sunaltaz = coordinates.get_sun(self.starttime+timedelta(seconds = self.resolution*i)).transform_to(altazframe)

            position.append([self.sunaltaz.az.degree,self.sunaltaz.alt.degree])
            distance.append(self.sunaltaz.distance.meter)

        self.az = [x[0] for x in position]
        self.alt = [x[1] for x in position]
        self.distance = distance

        moonposition = []
        moondistance = []

        for i in range(int(self.length/self.resolution)):

            altazframe = coordinates.AltAz(obstime = self.starttime+timedelta(seconds = self.resolution*i), location = self.location)
            self.moonaltaz = coordinates.get_body("moon", self.starttime+timedelta(seconds = self.resolution*i)).transform_to(altazframe)

            moonposition.append([self.moonaltaz.az.degree,self.moonaltaz.alt.degree])
            moondistance.append(self.moonaltaz.distance.meter)

        self.moonaz = [x[0] for x in moonposition]
        self.moonalt = [x[1] for x in moonposition]
        self.moondistance = moondistance


'''
GRAVITATIONAL WAVES (GWs)
'''

class GravWave:
    def __init__(
            self,
            frequency: float | int,
            strain: Decimal | float | int,
            phase_offset: float | int = np.pi / 2
        ) -> None:

        self.frequency = frequency
        self.strain = strain
        self.phase_offset = phase_offset

        """
        Stores a gravitational wave with a (dimensionless) strain of form
        s(t) = strain * sin(2 * Pi * frequency * t + phase_offset)
        """


def GW_strain(
        gw: GravWave,
        time: float | int | Decimal
    ) -> Decimal:

    """
    Calculates the change in strain of a GW as a function of time.
    """

    argument = Decimal(2) * pi() * Decimal(gw.frequency) * Decimal(time) + Decimal(gw.phase_offset)
    return Decimal(gw.strain) * Decimal(sin(argument % (Decimal(2) * pi())))

zeroGW = GravWave(frequency = 1, strain = 0, phase_offset = 0)

'''
Tungsten Test Mass Geometries
'''


class StanfordMass:
    def __init__(
        self,
        name: str = "Stanford",
        height: float | int = 0.03,                         # in meters
        inner_radius: float | int = 0.068,                  # in meters
        outer_radius: float | int = 0.078,                  # in meters
        final_integration_angle: float | int = 2.97         # rad (corresponds to 170°)
    ) -> None:

        self.name = name
        self.height = height                                        # Height of the ring in meters
        self.inner_radius = inner_radius                            # Inner radius of the ring in meters
        self.outer_radius = outer_radius                            # Outer radius of the ring in meters
        self.final_integration_angle = final_integration_angle      # Angular extent of the ring segment in rad


class StanfordMassSimplified:
    def __init__(
        self,
        name: str = "StanfordSimplified",
        mean_radial_distance: float | int = 0.073,      # mean radial distance
        mass: float | int = 1.25,                       # total mass in kg
    ) -> None:

        self.name = name
        self.mean_radial_distance = mean_radial_distance        # Radius of the thin ring around the baseline in meters
        self.mass = mass                                        # Total mass of the ring in kg


class TestMass:
    def __init__(
        self,
        experiment: StanfordMass | StanfordMassSimplified,
        radial_distance: float | int,
        mass_density: float | int = 19.25 * 10 ** 3
    ) -> None:

        self.experiment = experiment     # TestMass used in the corresponding experiment
        self.radial_distance = radial_distance                          # Radial distance from the reference point (usually center of mass) to the baseline
        self.mass_density = mass_density                                # Mass density of tungsten in kg / m^3 (StanfordMass only, the simplified ring has its total mass)

        # The simplified ring is a thin ring centred on the baseline. Its potential on the axis is - G M / sqrt(z^2 + R^2); off the axis, it would need elliptic integrals.
        # A radial distance would be ignored silently, so it is rejected. StanfordMass takes it into account.
        if isinstance(experiment, StanfordMassSimplified) and radial_distance != 0:
            raise ValueError(
                f"StanfordMassSimplified is a thin ring centred on the baseline and has no radial distance (given: {radial_distance} m). Use StanfordMass for a mass off the axis."
            )


class GGN_Noise_Model:
    def __init__(
        self,
        type: str | None = None,                 # Noise level: "NLNM" or "NHNM" (models "underground" and "seismic"), "low" or "high" (model "atmospheric")
        frequencies: np.ndarray | None = None,
        rng: int | None = None,
        model: str = "underground"               # "underground", "seismic" or "atmospheric", see gravity_gradient_noise_potential
    ) -> None:

        self.type = type
        self.frequencies = frequencies
        # Seed of the random GGN phases. It fixes the noise realization: The modes are drawn once per realization (see ggn_modes) and used at every evaluation of the potential.
        self.rng = rng if rng is not None else int(np.random.default_rng().integers(2**32))
        self.model = model


# END OF FILE
