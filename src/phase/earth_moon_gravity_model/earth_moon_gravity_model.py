'''
Created on Fri Jan 22 19:54:09 2021
@author: Dennis Schlippert

Changed on Oct 7 25
@ Michael
.get_moon() function is depreacated
Functionalities adapted for long term AIF analyses

Changed on Nov 6 26
@ Michael
Exact vertical tidal acceleration and tidal potential of the Sun and the Moon for a spherical Earth.

Geometry: The observer sits on a spherical Earth with radius E_r, the local vertical points away from the center of the Earth. A body (Sun or Moon) is seen at the topocentric
distance d under the altitude angle alt. Relative to the center of the Earth, the body then has the distance D with D^2 = R^2 + 2 R d sin(alt) + d^2, and its position has the component
D_v = R + d sin(alt) along the local vertical.

The tidal acceleration is the attraction of the body at the observer minus the one at the center of the Earth (which falls freely towards the body).
To leading order in R / D, its vertical component is G M r / D^3 (3 cos^2(psi) - 1) upwards (psi: geocentric zenith angle of the body, cos(psi) ≈ sin(alt)),
i.e. +2 G M R / D^3 for a body in the zenith and - G M R / D^3 at the horizon. The functions below use the exact expressions instead of this expansion.

Sign conventions of the PHASE code:
- acceleration() returns the value that is added to g (add_gravity in numerical_phases): Positive values increase g, i.e. the tidal acceleration points downwards.
  A body in the zenith therefore gives a negative value.
- The potentials are potential energies per mass as in model_phi, i.e. their derivative with respect to the height is the value of acceleration().
  They equal - W with the tide generating potential W = G M R^2 / D^3 (3 cos^2(psi) - 1) / 2 + O(R^3 / D^4) of geodesy.
All time series are mean-free, i.e. only the variation of the tides is returned.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G, hour

from phase.classes import location

M_sun = 1.99e30
M_moon = 7.35e22
E_r = 6.38e6


def plot(self: location) -> None:
    plt.figure(f"{self.name}, Longitude {self.location_lon:.2f} Latitude {self.location_lat:.2f}")
    plt.clf()
    x = np.linspace(0,self.length/hour,int(self.length/self.resolution))
    #plt.plot(x,self.az,label="Sun azimuth")
    plt.plot(x,self.alt,label = "Sun altitude")
    #plt.plot(x,self.moonaz,label="Moon azimuth")
    plt.plot(x,self.moonalt,label = "Moon Altitude")
    plt.grid(True)
    plt.legend(loc = "best")


def _geocentric_body_position(
        distance: np.ndarray,
        altitude: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

    """ Distance D of the body from the center of the Earth and the component D_v of its geocentric position along the local vertical of the observer. """
    sin_alt = np.sin(np.radians(altitude))
    vertical_component = E_r + distance * sin_alt
    geocentric_distance = np.sqrt(E_r ** 2 + 2 * E_r * distance * sin_alt + distance ** 2)

    return geocentric_distance, vertical_component


def _distance_to_body(
        geocentric_distance: np.ndarray,
        vertical_component: np.ndarray,
        radius: float
    ) -> np.ndarray:

    """ Distance between the body and a point on the local vertical of the observer at the distance radius from the center of the Earth. """
    return np.sqrt(geocentric_distance ** 2 + radius ** 2 - 2 * radius * vertical_component)


def tidal_acceleration(
        mass: float,
        distance: float | list[float] | np.ndarray,
        altitude: float | list[float] | np.ndarray,
        height: float = 0
    ) -> np.ndarray:

    """ Vertical tidal acceleration of a body with the given mass, seen at the topocentric distance (in m) and altitude (in degree), at the height above the observer (in m).
    Returned as the value that is added to g: Positive values point downwards and increase g.
    It is the radial derivative of tidal_potential: G M (r - D_v) / rho^3 + G M D_v / D^3 with the observer radius r = E_r + height and the distance rho between the point and the body. """

    geocentric_distance, vertical_component = _geocentric_body_position(distance = np.asarray(distance, dtype = float), altitude = np.asarray(altitude, dtype = float))
    radius = E_r + height
    distance_to_body = _distance_to_body(geocentric_distance = geocentric_distance, vertical_component = vertical_component, radius = radius)

    return G * mass * ((radius - vertical_component) / distance_to_body ** 3 + vertical_component / geocentric_distance ** 3)


def tidal_potential(
        mass: float,
        distance: float | list[float] | np.ndarray,
        altitude: float | list[float] | np.ndarray,
        height: float = 0
    ) -> np.ndarray:

    """ Tidal potential energy per mass of a body with the given mass, seen at the topocentric distance (in m) and altitude (in degree), at the height above the observer (in m).
    It is the potential - G M / rho of the body, minus its value G M / D and its gradient at the center of the Earth (the Earth falls freely towards the body):
    - G M / rho + G M / D + G M r D_v / D^3. To leading order this is - G M r^2 / D^3 (3 cos^2(psi) - 1) / 2, i.e. minus the tide generating potential of geodesy.
    The first two terms are combined to G M (r^2 - 2 r D_v) / (rho D (rho + D)), which avoids the cancellation of the large potentials G M / rho and G M / D. """

    geocentric_distance, vertical_component = _geocentric_body_position(distance = np.asarray(distance, dtype = float), altitude = np.asarray(altitude, dtype = float))
    radius = E_r + height
    distance_to_body = _distance_to_body(geocentric_distance = geocentric_distance, vertical_component = vertical_component, radius = radius)

    return G * mass * (
        (radius ** 2 - 2 * radius * vertical_component) / (distance_to_body * geocentric_distance * (distance_to_body + geocentric_distance))
        + radius * vertical_component / geocentric_distance ** 3
    )


def _mean_free(signal: np.ndarray) -> np.ndarray:
    return signal - np.mean(signal)


def acceleration(self: location) -> np.ndarray:

    """ Variation of the vertical tidal acceleration of the Sun and the Moon over the time series of the location, as it is added to g (positive values increase g). """
    return _mean_free(signal = tidal_acceleration(mass = M_sun, distance = self.distance, altitude = self.alt) + tidal_acceleration(mass = M_moon, distance = self.moondistance, altitude = self.moonalt))


def moon_potential(self: location) -> np.ndarray:

    """ Variation of the tidal potential (energy per mass, as in model_phi) of the Moon over the time series of the location. """
    return _mean_free(signal = tidal_potential(mass = M_moon, distance = self.moondistance, altitude = self.moonalt))


def sun_potential(self: location) -> np.ndarray:

    """ Variation of the tidal potential (energy per mass, as in model_phi) of the Sun over the time series of the location. """
    return _mean_free(signal = tidal_potential(mass = M_sun, distance = self.distance, altitude = self.alt))


def potential(self: location) -> np.ndarray:

    """ Variation of the tidal potential (energy per mass, as in model_phi) of the Sun and the Moon over the time series of the location. """
    return _mean_free(signal = tidal_potential(mass = M_sun, distance = self.distance, altitude = self.alt) + tidal_potential(mass = M_moon, distance = self.moondistance, altitude = self.moonalt))

# END OF FILE
