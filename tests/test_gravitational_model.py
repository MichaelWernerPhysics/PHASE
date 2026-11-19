'''
@author: Michael

Tests for the gravitational model (src/phase/gravitational_model/gravitational_model.py).
Convention of the model: phi(z) = g z - Gamma / 2 z^2 and model_g(z) = d phi / dz, i.e. g > 0 points downwards.
'''

import math
from collections.abc import Callable
from decimal import Decimal

import numpy as np
import pytest
import scipy.constants as scp

import phase.gravitational_model.gravitational_model as gm
from phase.classes import (
    GGN_Noise_Model,
    StanfordMassSimplified,
    TestMass as TungstenTestMass,  # Renamed, so that pytest does not try to collect it as a test class.
)
from phase.gravitational_model import (
    DEFAULT_G,
    DEFAULT_GRAVITY_GRADIENT,
    model_g,
    model_phi,
)
from phase.gravitational_model.GGN_noise_model import (
    PARAMS,
    gravity_gradient_noise_potential,
    make_modes,
)

MODELS = ["homogeneous", "constant gradient"]


def numerical_derivative(
        function: Callable[[Decimal], Decimal],
        h: Decimal,
        eps: Decimal = Decimal("1e-8")
    ) -> Decimal:

    return (function(h + eps) - function(h - eps)) / (2 * eps)


# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Default values ~~~
# ~~~~~~~~~~~~~~~~~~~~~~

def test_default_values_are_standard_gravity_and_free_air_gradient():
    assert DEFAULT_G == Decimal("9.80665")                  # m/s^2
    assert DEFAULT_GRAVITY_GRADIENT == Decimal("3.086e-6")  # 1/s^2


# ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ model_g/model_phi ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("h", [Decimal(0), Decimal(4), Decimal(-3)])
def test_homogeneous_g_is_constant(h):
    assert model_g(h = h, ideal = "homogeneous", absolute_time = Decimal(0)) == DEFAULT_G
    assert model_phi(h = h, ideal = "homogeneous", absolute_time = Decimal(0)) == DEFAULT_G * h


@pytest.mark.parametrize("h", [Decimal(0), Decimal(4), Decimal(-3)])
def test_constant_gradient_g_is_linear(h):
    assert model_g(h = h, ideal = "constant gradient", absolute_time = Decimal(0)) == DEFAULT_G - DEFAULT_GRAVITY_GRADIENT * h
    assert model_phi(h = h, ideal = "constant gradient", absolute_time = Decimal(0)) == DEFAULT_G * h - DEFAULT_GRAVITY_GRADIENT / 2 * h ** 2


''' g and the gravity gradient are parameters of the model. The gradient only enters the model "constant gradient". '''
@pytest.mark.parametrize("model", MODELS)
def test_g_and_gradient_are_parameters(model):
    h, g, gradient = Decimal(3), Decimal("9.812"), Decimal("2.5e-6")
    expected_gradient = gradient if model == "constant gradient" else Decimal(0)

    assert model_g(h = h, ideal = model, absolute_time = Decimal(0), g = g, gravity_gradient = gradient) == g - expected_gradient * h
    assert model_phi(h = h, ideal = model, absolute_time = Decimal(0), g = g, gravity_gradient = gradient) == g * h - expected_gradient / 2 * h ** 2


def test_model_type_is_case_insensitive():
    assert model_g(h = Decimal(2), ideal = "Constant Gradient", absolute_time = Decimal(0)) == model_g(h = Decimal(2), ideal = "constant gradient", absolute_time = Decimal(0))


''' The acceleration has to be the gradient of the potential, otherwise trajectories and propagation phases are inconsistent. '''
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("acceleration", [Decimal(0), Decimal("1e-6")])
@pytest.mark.parametrize("h", [Decimal("0.7"), Decimal("3.3")])
def test_model_g_is_gradient_of_model_phi(model, acceleration, h):
    derivative = numerical_derivative(function = lambda z: model_phi(h = z, ideal = model, absolute_time = Decimal(0), additional_global_acceleration = acceleration), h = h)

    assert abs(derivative - model_g(h = h, ideal = model, absolute_time = Decimal(0), additional_global_acceleration = acceleration)) < Decimal("1e-15")


@pytest.mark.parametrize("model", MODELS)
def test_additional_acceleration_adds_to_g(model):
    acceleration = Decimal("1e-6")
    shift = model_g(h = Decimal(2), ideal = model, absolute_time = Decimal(0), additional_global_acceleration = acceleration) - model_g(h = Decimal(2), ideal = model, absolute_time = Decimal(0))

    assert shift == acceleration


@pytest.mark.parametrize("function", [model_g, model_phi])
@pytest.mark.parametrize("model", ["flat earth", "non trivial"])
def test_unknown_model_type_is_rejected(function, model):
    with pytest.raises(ValueError, match = "constant gradient"):
        function(h = Decimal(1), ideal = model, absolute_time = Decimal(0))


def test_test_mass_without_heights_is_rejected():
    with pytest.raises(AttributeError):
        model_phi(h = Decimal(1), ideal = "homogeneous", absolute_time = Decimal(0), test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0))


def test_only_tungsten_without_test_mass_is_zero():
    assert model_phi(h = Decimal(1), ideal = "homogeneous", absolute_time = Decimal(0), only_tungsten = True) == 0
    assert model_g(h = Decimal(1), ideal = "homogeneous", absolute_time = Decimal(0), only_tungsten = True) == 0


def test_test_mass_potential_is_added():
    test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0)
    with_mass = model_phi(h = Decimal("0.05"), ideal = "homogeneous", absolute_time = Decimal(0), test_mass = test_mass, tungsten_positions = [0.0])
    without_mass = model_phi(h = Decimal("0.05"), ideal = "homogeneous", absolute_time = Decimal(0))

    experiment = test_mass.experiment
    assert isinstance(experiment, StanfordMassSimplified)
    expected = - scp.G * experiment.mass / math.sqrt(0.05 ** 2 + experiment.mean_radial_distance ** 2)

    assert float(with_mass - without_mass) == pytest.approx(expected, rel = 1e-6)


def test_ggn_potential_is_added():
    noise_model = GGN_Noise_Model(type = "NLNM", frequencies = np.linspace(0.1, 1, 10), rng = 3, model = "seismic")
    h, t = Decimal(2), Decimal("0.4")
    f, xi, phi = make_modes(f = noise_model.frequencies, type = noise_model.type, rng = noise_model.rng, model = noise_model.model)

    ggn = model_phi(h = h, ideal = "homogeneous", absolute_time = t, GGN_noise = noise_model) - model_phi(h = h, ideal = "homogeneous", absolute_time = t)

    assert float(ggn) == pytest.approx(gravity_gradient_noise_potential(z = h, t = t, f = f, xi = xi, phi = phi, params = PARAMS, model = "seismic"), rel = 1e-12)


''' The modes of a GGN realization are computed once and reused at every evaluation of the potential: The potential is evaluated at every time step of numerical_phases. '''
def test_ggn_modes_are_computed_once_per_realization(monkeypatch):
    calls: list[int] = []

    def counting_make_modes(**kwargs):
        calls.append(1)
        return make_modes(**kwargs)

    monkeypatch.setattr(gm, "make_modes", counting_make_modes)
    gm._cached_ggn_modes.cache_clear()
    noise_model = GGN_Noise_Model(type = "NLNM", frequencies = np.linspace(0.1, 1, 10), rng = 11, model = "seismic")

    values = [model_phi(h = Decimal(h), ideal = "homogeneous", absolute_time = Decimal("0.1"), GGN_noise = noise_model) for h in (1, 2, 3)]
    assert len(calls) == 1

    # A different seed is a different realization.
    other = GGN_Noise_Model(type = "NLNM", frequencies = np.linspace(0.1, 1, 10), rng = 12, model = "seismic")
    assert model_phi(h = Decimal(1), ideal = "homogeneous", absolute_time = Decimal("0.1"), GGN_noise = other) != values[0]
    assert len(calls) == 2


# ~~~~~~~~~~~~~~~~~~~~~
# ~~~ Mirror jitter ~~~
# ~~~~~~~~~~~~~~~~~~~~~

def test_displacement_noise_without_phases_is_zero():
    assert gm.create_displacement_noise_at_time(random_phases = None, time = 1.0) == 0


def test_displacement_noise_is_bounded_by_spectrum():
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * np.pi, len(gm.frequencies_and_amplitudes))
    bound = sum(amp / (2 * np.pi * freq) ** 2 for freq, amp, _ in gm.frequencies_and_amplitudes)

    assert gm.create_displacement_noise_at_time(random_phases = np.zeros(len(phases)), time = 0.0) == 0
    for time in [0.0, 0.3, 1.7]:
        assert abs(gm.create_displacement_noise_at_time(random_phases = phases, time = time)) <= bound


# END OF FILE
