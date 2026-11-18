'''
@author: Michael

Tests for the gravity gradient noise (GGN) model (src/phase/gravitational_model/GGN_noise_model.py).
'''

from decimal import Decimal

import numpy as np
import pytest

import phase.gravitational_model.GGN_noise_model as ggn

FREQUENCIES = np.linspace(0.1, 1.0, 10)


def single_mode(
        frequency: float = 0.5,
        amplitude: float = 10 ** -9,
        phase: float = 0.3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    return np.array([frequency]), np.array([amplitude]), np.array([phase])


def evaluate(
        z: float,
        t: float,
        f: np.ndarray,
        xi: np.ndarray,
        phi: np.ndarray,
        model: str,
        z_offset: float = 10.0
    ) -> float:

    """ The underground model needs a Decimal height, since it is evaluated as Decimal(z_offset) - z. """
    height = Decimal(z) if model == "underground" else z
    return ggn.gravity_gradient_noise_potential(z = height, t = t, f = f, xi = xi, phi = phi, z_offset = z_offset, model = model)


# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Noise spectra ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("noise_type", ["NLNM", "NHNM"])
def test_peterson_model_is_sorted_and_positive(noise_type):
    f, asd = ggn._peterson_asd(type = noise_type)

    assert np.all(np.diff(f) > 0)
    assert np.all(asd > 0)
    assert f.min() == pytest.approx(1e-5)
    assert f.max() == pytest.approx(10.0)


def test_low_noise_model_is_below_high_noise_model():
    f = np.logspace(-3, 1, 50)

    assert np.all(ggn._interp_asd(f_query = f, type = "NLNM") < ggn._interp_asd(f_query = f, type = "NHNM"))


def test_low_pressure_noise_is_below_high_pressure_noise(tmp_path, monkeypatch):
    # The pressure spectra are found next to the module, independent of the working directory.
    monkeypatch.chdir(tmp_path)
    _, psd_low = ggn._bowman_psd("low")
    _, psd_high = ggn._bowman_psd("high")

    assert np.median(psd_low) < np.median(psd_high)
    assert np.all(psd_low > 0) and np.all(psd_high > 0)


@pytest.mark.parametrize("loader, noise_type", [(ggn._peterson_asd, "medium"), (ggn._bowman_psd, "NLNM")])
def test_unknown_noise_type_is_rejected(loader, noise_type):
    with pytest.raises(ValueError):
        loader(noise_type)


def test_interpolation_reproduces_grid_and_vanishes_outside():
    f, asd = ggn._peterson_asd(type = "NLNM")

    assert ggn._interp_asd(f_query = f[100:110], type = "NLNM") == pytest.approx(asd[100:110], rel = 1e-10)
    assert np.all(ggn._interp_asd(f_query = [10 ** -9, 10 ** 3], type = "NLNM") == 0)


def test_local_bandwidth():
    assert ggn._local_bandwidth(f = [0.1, 0.2, 0.3, 0.4]) == pytest.approx([0.05, 0.1, 0.1, 0.05])
    # Unsorted input: the bandwidths stay at the position of their frequency.
    assert ggn._local_bandwidth(f = [0.3, 0.1, 0.2, 0.4]) == pytest.approx([0.1, 0.05, 0.1, 0.05])
    assert ggn._local_bandwidth(f = [0.7]) == pytest.approx([0.7])


# ~~~~~~~~~~~~~~~~~~~
# ~~~ Mode tuples ~~~
# ~~~~~~~~~~~~~~~~~~~

def test_seismic_mode_amplitude_is_asd_times_root_bandwidth():
    _, xi, phi = ggn.make_modes(f = FREQUENCIES, type = "NLNM", rng = 1)

    assert xi == pytest.approx(ggn._interp_asd(f_query = FREQUENCIES, type = "NLNM") * np.sqrt(ggn._local_bandwidth(f = FREQUENCIES)))
    assert np.all((0 <= phi) & (phi < 2 * np.pi))


''' Atmospheric model: Bowman PSD in Pa^2/Hz, amplitudes in bar, xi = sqrt(PSD df) * 1e-5. '''
def test_atmospheric_mode_amplitude_is_pressure_in_bar():
    _, xi, _ = ggn.make_modes(f = FREQUENCIES, type = "low", rng = 1, model = "atmospheric")

    expected = np.sqrt(ggn._interp_asd(f_query = FREQUENCIES, type = "low", model = "atmospheric") * ggn._local_bandwidth(f = FREQUENCIES)) * 10 ** -5

    assert xi == pytest.approx(expected)


def test_modes_are_reproducible_with_seed():
    _, _, phi_1 = ggn.make_modes(f = FREQUENCIES, type = "NLNM", rng = 42)
    _, _, phi_2 = ggn.make_modes(f = FREQUENCIES, type = "NLNM", rng = 42)
    _, _, phi_3 = ggn.make_modes(f = FREQUENCIES, type = "NLNM", rng = 43)

    assert np.array_equal(phi_1, phi_2)
    assert not np.array_equal(phi_1, phi_3)


def test_given_phases_and_bandwidth_are_used():
    phases = np.linspace(0, 1, len(FREQUENCIES))
    _, xi, phi = ggn.make_modes(f = FREQUENCIES, type = "NLNM", df = 0.01, phi = phases)

    assert np.array_equal(phi, phases)
    assert xi == pytest.approx(ggn._interp_asd(f_query = FREQUENCIES, type = "NLNM") * 0.1)


@pytest.mark.parametrize("kwargs", [dict(df = [0.1, 0.2]), dict(phi = [0.1, 0.2])])
def test_mode_shape_mismatch_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ggn.make_modes(f = FREQUENCIES, type = "NLNM", **kwargs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ GGN potential ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

MODELS = ["underground", "seismic", "atmospheric"]


@pytest.mark.parametrize("model", MODELS)
def test_potential_is_linear_in_amplitude(model):
    f, xi, phi = ggn.make_modes(f = FREQUENCIES, type = "NLNM", rng = 1)

    assert evaluate(z = 1.0, t = 0.3, f = f, xi = 2 * xi, phi = phi, model = model) == pytest.approx(2 * evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = model), rel = 1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_single_mode_is_periodic_in_time(model):
    f, xi, phi = single_mode()

    assert evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = model) == pytest.approx(evaluate(z = 1.0, t = 0.3 + 1 / f[0], f = f, xi = xi, phi = phi, model = model), rel = 1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_half_period_phase_shift_flips_sign(model):
    f, xi, phi = single_mode()

    assert evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi + np.pi, model = model) == pytest.approx(- evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = model), rel = 1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_zero_frequency_modes_are_ignored(model):
    assert evaluate(z = 1.0, t = 0.3, f = np.array([0.0]), xi = np.array([1.0]), phi = np.array([0.0]), model = model) == 0.0


''' Above ground, the seismic GGN decays as exp(- omega z / c_H) with the Rayleigh wave speed c_H. '''
def test_seismic_potential_decays_with_height():
    f, xi, phi = single_mode()
    decay = np.exp(- 2 * np.pi * f[0] * 3.0 / ggn.PARAMS["cH"])

    assert evaluate(z = 4.0, t = 0.3, f = f, xi = xi, phi = phi, model = "seismic") == pytest.approx(decay * evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = "seismic"), rel = 1e-12)


''' The underground model only depends on the depth z_offset - z below the surface. '''
def test_underground_potential_depends_on_depth():
    f, xi, phi = single_mode()

    assert evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = "underground", z_offset = 10.0) == pytest.approx(evaluate(z = 3.0, t = 0.3, f = f, xi = xi, phi = phi, model = "underground", z_offset = 12.0), rel = 1e-12)


def test_unknown_model_is_rejected():
    f, xi, phi = single_mode()
    with pytest.raises(ValueError):
        evaluate(z = 1.0, t = 0.3, f = f, xi = xi, phi = phi, model = "lunar")


def test_mode_arrays_must_have_same_shape():
    with pytest.raises(ValueError):
        evaluate(z = 1.0, t = 0.3, f = np.array([0.1, 0.2]), xi = np.array([1.0]), phi = np.array([0.0]), model = "seismic")


# END OF FILE
