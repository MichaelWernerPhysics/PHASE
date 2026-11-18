from decimal import Decimal

from numpy import pi, sin

from phase.white_noise import white_noise_lines

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Laser Phase Noise Model ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" The laser phase noise is white noise with a constant one-sided PSD in a frequency band, represented by one sinusoid per frequency line (see white_noise_lines).
numerical_phases includes it for interferometers with "Laser Phase" in their noise list and takes one random phase per line (random_phases_laser_noise),
e.g. numpy.random.default_rng(seed).uniform(0, 2 pi, len(frequencies_and_noise)).

The default of 1e-12 rad^2/Hz (1 µrad/sqrt(Hz)) between 0.1 Hz and 100 Hz is an example. Change it with set_white_phase_noise before the calculation. """

DEFAULT_PHASE_NOISE_PSD = 1e-12         # One-sided phase noise PSD in rad^2/Hz
DEFAULT_MIN_FREQUENCY = 0.1             # in Hz
DEFAULT_MAX_FREQUENCY = 100.0           # in Hz
DEFAULT_NUMBER_OF_LINES = 1000

# Frequencies in Hz, PSD in rad^2/Hz and (frequency in Hz, line amplitude in rad) of each line. Set by set_white_phase_noise.
frequency_list: list[float] = []
phase_noise_psd_list: list[float] = []
frequencies_and_noise: list[tuple[float, float]] = []


def set_white_phase_noise(
        psd: float = DEFAULT_PHASE_NOISE_PSD,
        min_frequency: float = DEFAULT_MIN_FREQUENCY,
        max_frequency: float = DEFAULT_MAX_FREQUENCY,
        number_of_lines: int = DEFAULT_NUMBER_OF_LINES
    ) -> None:

    """ Sets the laser phase noise to white noise with the one-sided PSD psd (in rad^2/Hz) between min_frequency and max_frequency (in Hz).
    The number of random phases passed to numerical_phases has to match number_of_lines. """

    global frequency_list, phase_noise_psd_list, frequencies_and_noise

    frequencies, amplitudes = white_noise_lines(psd = psd, min_frequency = min_frequency, max_frequency = max_frequency, number_of_lines = number_of_lines)
    frequency_list = frequencies
    phase_noise_psd_list = [psd] * number_of_lines
    frequencies_and_noise = list(zip(frequencies, amplitudes, strict = True))


set_white_phase_noise()


def phase_noise(phase_noise_model_phases, time):

    """ Laser phase noise in rad at the given time for the random phases of the lines. Without random phases (None), the laser has no phase noise. """

    noise_value = Decimal(0)
    if phase_noise_model_phases is None:
        return noise_value

    else:
        sample_range = min(len(phase_noise_model_phases), len(frequencies_and_noise))
        for index in range(sample_range):
            freq = frequencies_and_noise[index][0]
            noise = frequencies_and_noise[index][1]
            noise_value += Decimal(noise * sin(2 * pi * freq * float(time) + phase_noise_model_phases[index]))
        return noise_value


# END OF FILE
