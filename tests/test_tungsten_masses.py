'''
@author: Michael

Tests for the gravitational potential and acceleration of the tungsten test masses (src/phase/gravitational_model/tungsten_masses.py).
The "acceleration" in this module is d phi / dz, i.e. the same convention as model_g (positive values pull the atoms downwards).
'''

from pathlib import Path
from typing import Any, cast

import pytest
import scipy.constants as scp

import phase.gravitational_model.tungsten_masses as tm  # Imported as module, since the functions test_mass_... would be collected as tests otherwise.
from phase.classes import (
    StanfordMass,
    StanfordMassSimplified,
    TestMass as TungstenTestMass,  # Renamed, so that pytest does not try to collect it as a test class.
)

G = scp.G                   # Gravitational constant used in tungsten_masses.py
rho = 19.25 * 10 ** 3       # Default density of tungsten in kg/m^3


def stanford() -> TungstenTestMass:
    return TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.0)


def stanford_simplified() -> TungstenTestMass:
    return TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0)


''' Total masses of the geometries in kg. '''
stanford_mass = rho * 0.03 * (0.078 ** 2 - 0.068 ** 2) / 2 * 2.97

ALL_MASSES = [stanford, stanford_simplified]


@pytest.mark.parametrize("test_mass", ALL_MASSES)
@pytest.mark.parametrize("z", [0.3, -0.25])
def test_acceleration_is_gradient_of_potential(test_mass, z):
    eps = 10 ** -4
    derivative = (tm.test_mass_potential(z = z + eps, test_mass = test_mass(), vertical_position_of_mass = 0.0) - tm.test_mass_potential(z = z - eps, test_mass = test_mass(), vertical_position_of_mass = 0.0)) / (2 * eps)

    assert tm.test_mass_acceleration(z = z, test_mass = test_mass(), vertical_position_of_mass = 0.0) == pytest.approx(derivative, rel = 1e-5)


''' Far away from the mass, the potential is the one of a point mass, - G M / d. '''
@pytest.mark.parametrize("test_mass, total_mass", [
    (stanford, stanford_mass),
    (stanford_simplified, 1.25),
])
def test_far_field_is_point_mass(test_mass, total_mass):
    distance = 20.0

    assert tm.test_mass_potential(z = distance, test_mass = test_mass(), vertical_position_of_mass = 0.0) == pytest.approx(- G * total_mass / distance, rel = 1e-3)


@pytest.mark.parametrize("test_mass", ALL_MASSES)
def test_potential_is_symmetric_around_center_of_mass(test_mass):
    assert tm.test_mass_potential(z = 0.2, test_mass = test_mass(), vertical_position_of_mass = 1.0) == pytest.approx(tm.test_mass_potential(z = 1.8, test_mass = test_mass(), vertical_position_of_mass = 1.0), rel = 1e-8)
    assert tm.test_mass_acceleration(z = 0.2, test_mass = test_mass(), vertical_position_of_mass = 1.0) == pytest.approx(- tm.test_mass_acceleration(z = 1.8, test_mass = test_mass(), vertical_position_of_mass = 1.0), rel = 1e-8)


''' Above the mass the atoms are pulled downwards (positive acceleration in the model_g convention) and the potential is attractive. '''
@pytest.mark.parametrize("test_mass", ALL_MASSES)
def test_mass_attracts_atoms(test_mass):
    assert tm.test_mass_potential(z = 0.3, test_mass = test_mass(), vertical_position_of_mass = 0.0) < 0
    assert tm.test_mass_acceleration(z = 0.3, test_mass = test_mass(), vertical_position_of_mass = 0.0) > 0
    assert tm.test_mass_acceleration(z = -0.3, test_mass = test_mass(), vertical_position_of_mass = 0.0) < 0


def test_unknown_geometry_is_rejected():
    with pytest.raises(ValueError):
        tm.test_mass_potential(z = 0.1, test_mass = TungstenTestMass(experiment = cast(Any, object()), radial_distance = 0.1), vertical_position_of_mass = 0.0)
    with pytest.raises(ValueError):
        tm.test_mass_acceleration(z = 0.1, test_mass = TungstenTestMass(experiment = cast(Any, object()), radial_distance = 0.1), vertical_position_of_mass = 0.0)


# ~~~~~~~~~~~~~~~~~~~~~
# ~~~ Lookup tables ~~~
# ~~~~~~~~~~~~~~~~~~~~~

''' Only geometries with a lookup table in the repository are tested here. Creating a new table takes 2 x 100000 triple integrals. '''
@pytest.mark.parametrize("test_mass", [stanford, stanford_simplified])
@pytest.mark.parametrize("z", [0.01, 0.05, 0.2, -0.13])
def test_fast_lookup_matches_exact_integral(test_mass, z):
    assert tm.tungsten_potential_fast(z = z, z_center = 0.0, test_mass = test_mass()) == pytest.approx(tm.test_mass_potential(z = z, test_mass = test_mass(), vertical_position_of_mass = 0.0), rel = 1e-6)
    assert tm.tungsten_acceleration_fast(z = z, z_center = 0.0, test_mass = test_mass()) == pytest.approx(tm.test_mass_acceleration(z = z, test_mass = test_mass(), vertical_position_of_mass = 0.0), rel = 1e-6)


def test_fast_lookup_is_shifted_to_mass_position():
    assert tm.tungsten_potential_fast(z = 1.05, z_center = 1.0, test_mass = stanford_simplified()) == pytest.approx(tm.tungsten_potential_fast(z = 0.05, z_center = 0.0, test_mass = stanford_simplified()), rel = 1e-12)


def test_multiple_blocks_superpose():
    centers = [0.0, 0.4]
    z = 0.15

    total_potential = tm.multiple_tungsten_blocks_potential_fast(z = z, z_centers = centers, test_mass = stanford_simplified())
    total_acceleration = tm.multiple_tungsten_blocks_acceleration_fast(z = z, z_centers = centers, test_mass = stanford_simplified())

    assert total_potential == pytest.approx(sum(tm.tungsten_potential_fast(z = z, z_center = center, test_mass = stanford_simplified()) for center in centers), rel = 1e-12)
    assert total_acceleration == pytest.approx(sum(tm.tungsten_acceleration_fast(z = z, z_center = center, test_mass = stanford_simplified()) for center in centers), rel = 1e-12)


def test_single_block_position_as_float():
    assert tm.multiple_tungsten_blocks_potential_fast(z = 0.15, z_centers = 0.4, test_mass = stanford_simplified()) == tm.multiple_tungsten_blocks_potential_fast(z = 0.15, z_centers = [0.4], test_mass = stanford_simplified())
    assert tm.multiple_tungsten_blocks_acceleration_fast(z = 0.15, z_centers = 0.4, test_mass = stanford_simplified()) == tm.multiple_tungsten_blocks_acceleration_fast(z = 0.15, z_centers = [0.4], test_mass = stanford_simplified())


def test_no_test_mass_gives_zero():
    assert tm.tungsten_potential_fast(z = 0.1, z_center = 0.0, test_mass = None) == 0
    assert tm.tungsten_acceleration_fast(z = 0.1, z_center = 0.0, test_mass = None) == 0


def test_lookup_table_is_cached():
    first = tm.initialize_test_mass(test_mass = stanford_simplified(), potential_or_acceleration = "Both")
    second = tm.initialize_test_mass(test_mass = stanford_simplified(), potential_or_acceleration = "Both")

    assert first is second


''' A lookup table is only valid for the parameters it was created with: radial distance, mass density and the geometry of the experiment. The same parameters give the same table,
also if a number is given as int instead of float. The in-memory cache of the interpolated tables uses the same key. '''
def test_lookup_table_path_depends_on_radial_distance():
    near = TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.0)
    far = TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.1)

    assert tm.test_mass_lookup_table_path(test_mass = near) != tm.test_mass_lookup_table_path(test_mass = far)


@pytest.mark.parametrize("other", [
    TungstenTestMass(experiment = StanfordMass(height = 0.04), radial_distance = 0.0),
    TungstenTestMass(experiment = StanfordMass(inner_radius = 0.06), radial_distance = 0.0),
    TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.0, mass_density = 18 * 10 ** 3),
])
def test_lookup_table_path_depends_on_all_parameters(other):
    reference = TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.0)

    assert tm.test_mass_lookup_table_path(test_mass = other) != tm.test_mass_lookup_table_path(test_mass = reference)
    assert tm._make_test_mass_cache_key(test_mass = other, potential_or_acceleration = "Both") != tm._make_test_mass_cache_key(test_mass = reference, potential_or_acceleration = "Both")


def test_lookup_table_path_does_not_depend_on_number_type():
    assert tm.test_mass_lookup_table_path(test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0)) == tm.test_mass_lookup_table_path(test_mass = TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.0))


''' The lookup tables of the default Stanford geometries are part of the package. They are found next to the module, independent of the working directory. '''
@pytest.mark.parametrize("test_mass", [stanford, stanford_simplified])
def test_lookup_tables_of_default_geometries_are_shipped(test_mass, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(tm.test_mass_lookup_table_path(test_mass = test_mass()))

    assert path.parent == Path(tm.__file__).resolve().parent
    assert path.is_file()


''' Lookup tables for new parameters are created in the cache directory (PHASE_CACHE_DIR, otherwise the cache directory of the user), not in the package:
Its directory may be read-only and is replaced by every installation. '''
def test_new_lookup_table_is_stored_in_the_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE_CACHE_DIR", str(tmp_path))
    test_mass = TungstenTestMass(experiment = StanfordMassSimplified(mass = 2.0), radial_distance = 0.0)
    path = Path(tm.test_mass_lookup_table_path(test_mass = test_mass))

    assert path.parent == tmp_path
    assert not path.exists()

    potential, _ = tm.initialize_test_mass(test_mass = test_mass)

    assert path.is_file()
    assert not (tm.LOOKUP_TABLE_DIRECTORY / path.name).exists()
    assert float(potential(0.5)) == pytest.approx(tm.test_mass_potential(z = 0.5, test_mass = test_mass, vertical_position_of_mass = 0), rel = 1e-9)


def test_cache_directory_is_not_in_the_package(monkeypatch):
    monkeypatch.delenv("PHASE_CACHE_DIR", raising = False)
    cache_directory = tm.lookup_table_cache_directory()

    assert cache_directory != tm.LOOKUP_TABLE_DIRECTORY
    assert tm.LOOKUP_TABLE_DIRECTORY not in cache_directory.parents


''' The simplified ring is centred on the baseline. Its potential has no radial distance, so a radial distance is rejected instead of being ignored. '''
def test_simplified_ring_rejects_radial_distance():
    with pytest.raises(ValueError, match = "centred on the baseline"):
        TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0.1)

    TungstenTestMass(experiment = StanfordMassSimplified(), radial_distance = 0)
    TungstenTestMass(experiment = StanfordMass(), radial_distance = 0.1)


# END OF FILE
