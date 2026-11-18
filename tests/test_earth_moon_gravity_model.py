'''
@author: Michael

Tests for the tidal model of the Sun and the Moon (src/phase/earth_moon_gravity_model/earth_moon_gravity_model.py).
The location objects are replaced by simple stand-ins with given altitudes and distances, so that no ephemeris calculation is needed.
'''

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy.constants import G, hour

import phase.earth_moon_gravity_model as em
from phase.classes import location

AU = 1.496 * 10 ** 11       # Sun distance in m
MOON_DISTANCE = 3.844 * 10 ** 8

def fake_location(
        sun_altitudes: list[float],
        moon_altitudes: list[float],
        sun_distance: float = AU,
        moon_distance: float = MOON_DISTANCE
    ) -> location:

    # Only the attributes used by acceleration are set, hence the cast.
    return cast(location, SimpleNamespace(
        name = "Test",
        location_lon = 9.7,
        location_lat = 52.4,
        length = len(sun_altitudes) * hour,
        resolution = hour,
        alt = list(sun_altitudes),
        distance = [sun_distance] * len(sun_altitudes),
        moonalt = list(moon_altitudes),
        moondistance = [moon_distance] * len(moon_altitudes),
    ))


def tidal_scale(mass: float, distance: float) -> float:
    return G * mass * em.earth_moon_gravity_model.E_r / distance ** 3


@pytest.mark.parametrize("function", [em.acceleration, em.potential, em.sun_potential, em.moon_potential])
def test_output_is_mean_free(function):
    signal = function(fake_location(sun_altitudes = [10, 30, 50, 20], moon_altitudes = [80, 40, 0, 60]))

    assert np.mean(signal) == pytest.approx(0, abs = 10 ** -12 * np.max(np.abs(signal)))


@pytest.mark.parametrize("function", [em.acceleration, em.potential, em.sun_potential, em.moon_potential])
def test_constant_geometry_gives_no_signal(function):
    # Scale of the signal if the geometry changes, to compare the rounding errors with.
    scale = np.max(np.abs(function(fake_location(sun_altitudes = [0, 90, 45], moon_altitudes = [90, 0, 45]))))

    assert np.all(np.abs(function(fake_location(sun_altitudes = [30] * 3, moon_altitudes = [45] * 3))) < 10 ** -12 * scale)


R = em.earth_moon_gravity_model.E_r
BODIES = [("sun", em.earth_moon_gravity_model.M_sun, AU), ("moon", em.earth_moon_gravity_model.M_moon, MOON_DISTANCE)]


def vertical_shape(altitude: float) -> float:
    return 3 * np.sin(np.radians(altitude)) ** 2 - 1


''' Vertical tide, added to g: - G M R / D^3 (3 sin^2(alt) - 1) to leading order, i.e. - 2 G M R / d^3 in the zenith (g decreases) and + G M R / d^3 at the horizon (g increases).
The exact expression deviates by corrections of order R / d (about 2 % for the Moon). The node at sin^2(alt) = 1/3 is left out, since the relative deviation is not defined there. '''
@pytest.mark.parametrize("altitude", [90, 60, 0, -45, -90])
@pytest.mark.parametrize("body, mass, distance", BODIES)
def test_tidal_acceleration_leading_order(body, mass, distance, altitude):
    expected = - tidal_scale(mass = mass, distance = distance) * vertical_shape(altitude = altitude)

    assert float(em.tidal_acceleration(mass = mass, distance = distance, altitude = altitude)) == pytest.approx(expected, rel = 4 * R / distance)


''' Tidal potential energy per mass (as model_phi): - G M R^2 / D^3 (3 sin^2(alt) - 1) / 2 to leading order, i.e. minus the tide generating potential W of geodesy. '''
@pytest.mark.parametrize("altitude", [90, 60, 0, -45, -90])
@pytest.mark.parametrize("body, mass, distance", BODIES)
def test_tidal_potential_leading_order(body, mass, distance, altitude):
    expected = - tidal_scale(mass = mass, distance = distance) * R * vertical_shape(altitude = altitude) / 2

    assert float(em.tidal_potential(mass = mass, distance = distance, altitude = altitude)) == pytest.approx(expected, rel = 4 * R / distance)


''' As for model_phi and add_gravity, the derivative of the potential with respect to the height is the acceleration that is added to g. '''
@pytest.mark.parametrize("altitude", [90, 60, 0, -45])
@pytest.mark.parametrize("body, mass, distance", BODIES)
def test_tidal_potential_derivative_is_acceleration(body, mass, distance, altitude):
    step = 10.0
    derivative = (em.tidal_potential(mass = mass, distance = distance, altitude = altitude, height = step) - em.tidal_potential(mass = mass, distance = distance, altitude = altitude, height = - step)) / (2 * step)

    assert float(derivative) == pytest.approx(float(em.tidal_acceleration(mass = mass, distance = distance, altitude = altitude)), rel = 1e-5)


''' Order of magnitude of the tides: In the zenith, the Sun reduces g by about 0.5 um/s^2 and the Moon by about 1.1 um/s^2 (2 G M R / d^3).
From the horizon into the zenith, g therefore changes by - 3 G M R / d^3 and the potential by - 3 G M R^2 / (2 d^3), up to corrections of order R / d. '''
@pytest.mark.parametrize("body, mass, distance", BODIES)
def test_tides_between_horizon_and_zenith(body, mass, distance):
    # The other body stays in the zenith (no variation); the body of interest moves from the horizon into the zenith.
    if body == "sun":
        location, body_potential = fake_location(sun_altitudes = [0, 90], moon_altitudes = [90, 90]), em.sun_potential
    else:
        location, body_potential = fake_location(sun_altitudes = [90, 90], moon_altitudes = [0, 90]), em.moon_potential

    acceleration = em.acceleration(location)
    potential = body_potential(location)

    assert acceleration[1] - acceleration[0] == pytest.approx(- 3 * tidal_scale(mass = mass, distance = distance), rel = 3 * R / distance)
    assert potential[1] - potential[0] == pytest.approx(- 3 * tidal_scale(mass = mass, distance = distance) * R / 2, rel = 3 * R / distance)


def test_potential_is_sum_of_sun_and_moon():
    location = fake_location(sun_altitudes = [10, 30, 50, 20], moon_altitudes = [80, 40, 0, 60])

    assert em.potential(location) == pytest.approx(em.sun_potential(location) + em.moon_potential(location), rel = 1e-12)


# END OF FILE
