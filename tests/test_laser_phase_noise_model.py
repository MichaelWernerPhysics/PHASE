'''
@author: Michael

Tests for the white noise models: the line representation (src/phase/white_noise.py), the laser phase noise (src/phase/laser_phase_noise_model/laser_phase_noise_model.py)
and the mirror jitter (src/phase/gravitational_model/gravitational_model.py).
'''

from decimal import Decimal

import numpy as np
import pytest

import phase.gravitational_model.gravitational_model as gm
import phase.laser_phase_noise_model.laser_phase_noise_model as lpn
from phase.laser_phase_noise_model import phase_noise
from phase.white_noise import white_noise_lines


@pytest.fixture(autouse = True)
def default_noise():
    """ Every test starts and ends with the default spectra, since set_white_phase_noise and set_white_mirror_jitter change the module state. """
    lpn.set_white_phase_noise()
    gm.set_white_mirror_jitter()
    yield
    lpn.set_white_phase_noise()
    gm.set_white_mirror_jitter()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Line representation ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' The lines sit in the centers of equally wide bins that cover the band, each line carries the power S df of its bin: a^2 / 2 = S df. '''
def test_white_noise_lines_cover_the_band():
    frequencies, amplitudes = white_noise_lines(psd = 4.0, min_frequency = 1.0, max_frequency = 2.0, number_of_lines = 4)

    assert frequencies == pytest.approx([1.125, 1.375, 1.625, 1.875])
    assert amplitudes == pytest.approx([np.sqrt(2 * 4.0 * 0.25)] * 4)


''' The variance of the noise is the sum of the line variances a^2 / 2 and equals the integral of the PSD over the band, independent of the number of lines. '''
@pytest.mark.parametrize("number_of_lines", [1, 10, 1000])
def test_total_variance_is_integrated_psd(number_of_lines):
    _, amplitudes = white_noise_lines(psd = 1e-12, min_frequency = 0.1, max_frequency = 100.0, number_of_lines = number_of_lines)

    assert sum(amplitude ** 2 / 2 for amplitude in amplitudes) == pytest.approx(1e-12 * 99.9)


@pytest.mark.parametrize("arguments", [
    dict(psd = -1.0, min_frequency = 0.1, max_frequency = 1.0, number_of_lines = 10),
    dict(psd = 1.0, min_frequency = 1.0, max_frequency = 1.0, number_of_lines = 10),
    dict(psd = 1.0, min_frequency = -0.1, max_frequency = 1.0, number_of_lines = 10),
    dict(psd = 1.0, min_frequency = 0.1, max_frequency = 1.0, number_of_lines = 0),
])
def test_invalid_white_noise_is_rejected(arguments):
    with pytest.raises(ValueError):
        white_noise_lines(**arguments)


# ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Laser phase noise ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~

def test_default_spectrum():
    assert len(lpn.frequency_list) == len(lpn.phase_noise_psd_list) == len(lpn.frequencies_and_noise) == lpn.DEFAULT_NUMBER_OF_LINES
    assert all(psd == lpn.DEFAULT_PHASE_NOISE_PSD for psd in lpn.phase_noise_psd_list)
    assert lpn.DEFAULT_MIN_FREQUENCY < min(lpn.frequency_list) < max(lpn.frequency_list) < lpn.DEFAULT_MAX_FREQUENCY


def test_set_white_phase_noise_changes_the_spectrum():
    lpn.set_white_phase_noise(psd = 1e-10, min_frequency = 1.0, max_frequency = 11.0, number_of_lines = 5)

    assert lpn.frequency_list == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0])
    assert [amplitude for _, amplitude in lpn.frequencies_and_noise] == pytest.approx([np.sqrt(2 * 1e-10 * 2.0)] * 5)


''' The spread of phase_noise over many random realizations is the integrated PSD. '''
@pytest.mark.slow
def test_noise_rms_is_integrated_psd():
    rng = np.random.default_rng(3)
    samples = [float(phase_noise(phase_noise_model_phases = rng.uniform(0, 2 * np.pi, len(lpn.frequencies_and_noise)), time = 0.37)) for _ in range(2000)]

    expected_rms = np.sqrt(lpn.DEFAULT_PHASE_NOISE_PSD * (lpn.DEFAULT_MAX_FREQUENCY - lpn.DEFAULT_MIN_FREQUENCY))
    assert np.std(samples) == pytest.approx(expected_rms, rel = 0.1)


def test_no_noise_without_phases():
    assert phase_noise(phase_noise_model_phases = None, time = 1.0) == 0


def test_noise_vanishes_at_zero_time_with_zero_phases():
    assert phase_noise(phase_noise_model_phases = [0.0] * len(lpn.frequencies_and_noise), time = 0.0) == 0


def test_noise_is_bounded_by_line_amplitudes():
    phases = np.random.default_rng(2).uniform(0, 2 * np.pi, len(lpn.frequencies_and_noise))
    bound = sum(amplitude for _, amplitude in lpn.frequencies_and_noise)

    for time in [0.0, 0.37, Decimal("1.2")]:
        noise = phase_noise(phase_noise_model_phases = phases, time = time)
        assert isinstance(noise, Decimal)
        assert abs(noise) <= bound


''' Only as many lines as random phases are given are summed up. '''
def test_number_of_lines_is_set_by_phases():
    frequency, amplitude = lpn.frequencies_and_noise[0]
    time, phase = 0.21, 0.4

    assert float(phase_noise(phase_noise_model_phases = [phase], time = time)) == pytest.approx(amplitude * np.sin(2 * np.pi * frequency * time + phase))


# ~~~~~~~~~~~~~~~~~~~~~
# ~~~ Mirror jitter ~~~
# ~~~~~~~~~~~~~~~~~~~~~

def test_default_mirror_spectrum():
    assert len(gm.frequencies_and_amplitudes) == gm.DEFAULT_NUMBER_OF_LINES
    assert all(asd == gm.DEFAULT_ACCELERATION_ASD for _, _, asd in gm.frequencies_and_amplitudes)
    assert sum(amplitude ** 2 / 2 for _, amplitude, _ in gm.frequencies_and_amplitudes) == pytest.approx(gm.DEFAULT_ACCELERATION_ASD ** 2 * (gm.DEFAULT_MAX_FREQUENCY - gm.DEFAULT_MIN_FREQUENCY))


''' Each acceleration line a sin(2 pi f t + phase) displaces the mirror by - a sin(2 pi f t + phase) / (2 pi f)^2. '''
def test_mirror_displacement_of_a_single_line():
    gm.set_white_mirror_jitter(acceleration_asd = 1e-6, min_frequency = 1.0, max_frequency = 3.0, number_of_lines = 1)
    (frequency, amplitude, _), = gm.frequencies_and_amplitudes
    time, phase = 0.3, 1.1

    assert frequency == pytest.approx(2.0)
    assert gm.create_displacement_noise_at_time(random_phases = [phase], time = time) == pytest.approx(- amplitude * np.sin(2 * np.pi * frequency * time + phase) / (2 * np.pi * frequency) ** 2)


def test_mirror_does_not_move_without_phases():
    assert gm.create_displacement_noise_at_time(random_phases = None, time = 1.0) == 0


# END OF FILE
