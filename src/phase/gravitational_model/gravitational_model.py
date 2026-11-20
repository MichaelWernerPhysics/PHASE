from collections.abc import Sequence
from decimal import Decimal
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy import pi, sin

from phase.classes import GGN_Noise_Model, TestMass
from phase.gravitational_model.GGN_noise_model import (
    PARAMS,
    gravity_gradient_noise_potential,
    make_modes,
)
from phase.gravitational_model.tungsten_masses import (
    multiple_tungsten_blocks_acceleration_fast,
    multiple_tungsten_blocks_potential_fast,
)
from phase.white_noise import white_noise_lines

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Gravitational model ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" The gravitational field along the vertical baseline (height h, pointing upwards) is either homogeneous or has a constant gradient:
    g(h) = g - Gamma h,     phi(h) = g h - Gamma / 2 h^2,
with the gravitational acceleration g (pointing downwards) and the vertical gravity gradient Gamma. phi is the potential energy per mass.
Local sources (tungsten test masses), gravity gradient noise and additional global accelerations (e.g. tides, see earth_moon_gravity_model) are added on top. """

# Standard gravity (CGPM 1901) in m/s^2 and the free-air gradient of the Earth in 1/s^2.
DEFAULT_G = Decimal("9.80665")
DEFAULT_GRAVITY_GRADIENT = Decimal("3.086e-6")

# Model of the field: "homogeneous" or "constant gradient". The functions accept any capitalization.
GravityModel = Literal["homogeneous", "constant gradient"]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Mirror jitter: vibration spectrum ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" Acceleration noise of the retro-reflection mirror. It is used for the mirror jitter ("Mirror Jitter" in the noise list of the interferometer).
The acceleration noise is white with a constant ASD in a frequency band, represented by one sinusoid per frequency line (see white_noise_lines).
numerical_phases takes one random phase per line (random_phases_mirror_jitter).

The default of 1e-8 m/s^2/sqrt(Hz) between 0.1 Hz and 100 Hz is an example. Change it with set_white_mirror_jitter before the calculation. """

DEFAULT_ACCELERATION_ASD = 1e-8         # Acceleration ASD in m/s^2/sqrt(Hz)
DEFAULT_MIN_FREQUENCY = 0.1             # in Hz
DEFAULT_MAX_FREQUENCY = 100.0           # in Hz
DEFAULT_NUMBER_OF_LINES = 1000

# (frequency in Hz, line amplitude of the acceleration in m/s^2, acceleration ASD in m/s^2/sqrt(Hz)) for each line. Set by set_white_mirror_jitter.
frequencies_and_amplitudes: list[tuple[float, float, float]] = []


def set_white_mirror_jitter(
        acceleration_asd: float = DEFAULT_ACCELERATION_ASD,
        min_frequency: float = DEFAULT_MIN_FREQUENCY,
        max_frequency: float = DEFAULT_MAX_FREQUENCY,
        number_of_lines: int = DEFAULT_NUMBER_OF_LINES
    ) -> None:

    """ Sets the acceleration noise of the mirror to white noise with the ASD acceleration_asd (in m/s^2/sqrt(Hz)) between min_frequency and max_frequency (in Hz).
    The number of random phases passed to numerical_phases has to match number_of_lines. """

    global frequencies_and_amplitudes

    frequencies, amplitudes = white_noise_lines(psd = acceleration_asd ** 2, min_frequency = min_frequency, max_frequency = max_frequency, number_of_lines = number_of_lines)
    frequencies_and_amplitudes = [(frequency, amplitude, acceleration_asd) for frequency, amplitude in zip(frequencies, amplitudes, strict = True)]


set_white_mirror_jitter()


def create_displacement_noise_at_time(random_phases, time):

    """ Displacement of the mirror in m at the given time. Each acceleration line a sin(2 pi f t + phase) displaces the mirror by - a sin(2 pi f t + phase) / (2 pi f)^2.
    Without random phases (None), the mirror does not move. """

    noise_value = 0
    if random_phases is None:
        return noise_value

    else:
        for index in range(len(frequencies_and_amplitudes)):
            freq = frequencies_and_amplitudes[index][0]
            amp = frequencies_and_amplitudes[index][1]
            noise_value += - amp * sin(2 * pi * freq * float(time) + random_phases[index]) / (2 * pi * freq) ** 2
        return noise_value


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Gravity gradient noise: modes of a realization ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@lru_cache(maxsize = 64)
def _cached_ggn_modes(
        frequencies: tuple[float, ...],
        noise_type: str | None,
        seed: int | None,
        model: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    modes = make_modes(f = np.array(frequencies, dtype = float), type = noise_type, rng = seed, model = model)
    for array in modes:
        array.setflags(write = False)  # The arrays are shared by all evaluations of the potential.
    return modes


def ggn_modes(
        GGN_noise: GGN_Noise_Model
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """ Frequencies, amplitudes and phases of the modes of a GGN realization, see make_modes. They only depend on the frequencies, the noise level, the seed and the model,
    so they are computed once per realization instead of at every evaluation of the potential. """
    frequencies = tuple(np.asarray(GGN_noise.frequencies, dtype = float).ravel().tolist())
    return _cached_ggn_modes(frequencies = frequencies, noise_type = GGN_noise.type, seed = GGN_noise.rng, model = GGN_noise.model)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Gravitational potential and acceleration ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_phi(
        # Height argument in m.
        h: Decimal | float | int,
        # Model of the field, either "homogeneous" or "constant gradient" (not case-sensitive).
        model: GravityModel,
        # Time argument is s.
        absolute_time: Decimal | float | int,
        # Additional global acceleration in m/s^2. Example: Change in g due to the Sun or the moon.
        additional_global_acceleration: Decimal = Decimal(0),
        # Type of test mass (if any)
        test_mass: TestMass | None = None,
        # If tungsten masses are included, this is a list of their heights in m. Example: [1, 6.3] for two blocks at 1m and 6.3m.
        tungsten_positions: Sequence[float] | float | None = None,
        # If True, only the potential of the test masses and the GGN is returned.
        only_tungsten: bool = False,
        # Gravity gradient noise. None means no GGN.
        GGN_noise: GGN_Noise_Model | None = None,
        # Reference height of the GGN model "underground" in m, see gravity_gradient_noise_potential.
        z_offset: float = 10.0,
        # Gravitational acceleration in m/s^2.
        g: Decimal = DEFAULT_G,
        # Vertical gravity gradient in 1/s^2. Only used for the model "constant gradient".
        gravity_gradient: Decimal = DEFAULT_GRAVITY_GRADIENT
    ) -> Decimal:

    """ Gravitational potential energy per mass in J/kg at the height h and the time absolute_time. """

    if GGN_noise is not None:
        f_modes, xi_modes, phi_modes = ggn_modes(GGN_noise = GGN_noise)

        ggn_pot = Decimal(gravity_gradient_noise_potential(
            z = h,
            t = absolute_time,
            f = f_modes,
            xi = xi_modes,
            phi = phi_modes,
            params = PARAMS,
            z_offset = z_offset,
            model = GGN_noise.model
        ))

    else:
        ggn_pot = Decimal(0)

    if (test_mass is None and tungsten_positions is not None) or (test_mass is not None and tungsten_positions is None):
        raise AttributeError("You can't specify a test mass configuration without specifying their heights or vice versa.")

    h = Decimal(h)
    absolute_time = Decimal(absolute_time)

    if test_mass is None:
        tungsten_potential = Decimal(0)  # in J/kg

    else:
        assert tungsten_positions is not None  # Guaranteed by the check above.
        tungsten_potential = Decimal(multiple_tungsten_blocks_potential_fast(
            z = float(h),
            z_centers = tungsten_positions,
            test_mass = test_mass
        ))  # in J/kg

    if only_tungsten:
        return tungsten_potential + ggn_pot

    if model.lower() == "homogeneous":
        return (g + additional_global_acceleration) * h + tungsten_potential + ggn_pot

    elif model.lower() == "constant gradient":
        return (g + additional_global_acceleration) * h - gravity_gradient / 2 * h ** 2 + tungsten_potential + ggn_pot

    else:
        raise ValueError(f"Unknown gravitational model {model!r}. It has to be either 'homogeneous' or 'constant gradient'.")


def model_g(
        # Height argument in m.
        h: Decimal,
        # Model of the field, either "homogeneous" or "constant gradient" (not case-sensitive).
        model: GravityModel,
        # Time argument is s.
        absolute_time: Decimal,
        # Additional global acceleration in m/s^2. Example: Change in g due to the Sun or the moon.
        additional_global_acceleration: Decimal = Decimal(0),
        # Type of test mass (if any)
        test_mass: TestMass | None = None,
        # If tungsten masses are included, this is a list of their heights in m.
        tungsten_positions: Sequence[float] | float | None = None,
        # If True, only the acceleration of the test masses is returned.
        only_tungsten: bool = False,
        # Gravitational acceleration in m/s^2.
        g: Decimal = DEFAULT_G,
        # Vertical gravity gradient in 1/s^2. Only used for the model "constant gradient".
        gravity_gradient: Decimal = DEFAULT_GRAVITY_GRADIENT
    ) -> Decimal:

    """ Gravitational acceleration in m/s^2 (positive values point downwards) at the height h and the time absolute_time. """

    if tungsten_positions is None:
        tungsten_acceleration = Decimal(0)    # in m/s^2

    else:
        tungsten_acceleration = Decimal(multiple_tungsten_blocks_acceleration_fast(
            z = float(h),
            z_centers = tungsten_positions,
            test_mass = test_mass
        ))    # in m/s^2

    if only_tungsten:
        return tungsten_acceleration

    if model.lower() == "homogeneous":
        return g + additional_global_acceleration + tungsten_acceleration

    elif model.lower() == "constant gradient":
        return g - gravity_gradient * h + additional_global_acceleration + tungsten_acceleration

    else:
        raise ValueError(f"Unknown gravitational model {model!r}. It has to be either 'homogeneous' or 'constant gradient'.")


# END OF FILE
