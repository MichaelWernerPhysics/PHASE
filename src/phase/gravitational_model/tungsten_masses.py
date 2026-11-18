from collections.abc import Sequence
from pathlib import Path

from numpy import array, cos, linspace, sqrt
from scipy import integrate
from scipy.interpolate import interp1d

from phase.classes import StanfordMass, StanfordMassSimplified, TestMass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Tungsten Test Masses ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The lookup tables are stored next to this file.
LOOKUP_TABLE_DIRECTORY = Path(__file__).resolve().parent

def test_mass_potential(
        z: float,
        test_mass: TestMass,
        vertical_position_of_mass: float,
    ) -> float:

    G = 6.674 * 10 ** -11  # Grav. constant in N m^2 / kg^2
    rho = test_mass.mass_density
    distance = test_mass.radial_distance
    experiment = test_mass.experiment

    def pot_integrand(zz, ww, rr):
        return - G * rho * rr / sqrt(distance ** 2 + rr ** 2 + 2 * distance * rr * cos(ww) + (z - zz) ** 2)

    if isinstance(experiment, StanfordMass):

        L = experiment.height
        R = experiment.outer_radius
        r = experiment.inner_radius
        phi_max = experiment.final_integration_angle

        return integrate.tplquad(pot_integrand, r, R, 0, phi_max, - L / 2 + vertical_position_of_mass, L / 2 + vertical_position_of_mass)[0]

    elif isinstance(experiment, StanfordMassSimplified):

        r = experiment.mean_radial_distance
        mass = experiment.mass

        return - G * mass / sqrt((vertical_position_of_mass - z) ** 2 + r ** 2)

    else:
        raise ValueError("Test Mass Configuration Unknown.")


def test_mass_acceleration(
        z: float,
        test_mass: TestMass,
        vertical_position_of_mass: float,
    ) -> float:

    G = 6.674 * 10 ** -11  # Grav. constant in N m^2 / kg^2
    rho = test_mass.mass_density
    distance = test_mass.radial_distance
    experiment = test_mass.experiment

    def acc_integrand(zz, ww, rr):
        return G * rho * rr * (z - zz) / sqrt(distance ** 2 + rr ** 2 + 2 * distance * rr * cos(ww) + (z - zz) ** 2) ** 3

    if isinstance(experiment, StanfordMass):

        L = experiment.height
        R = experiment.outer_radius
        r = experiment.inner_radius
        phi_max = experiment.final_integration_angle

        return integrate.tplquad(acc_integrand, r, R, 0, phi_max,  - L / 2 + vertical_position_of_mass, L / 2 + vertical_position_of_mass)[0]

    elif isinstance(experiment, StanfordMassSimplified):

        r = experiment.mean_radial_distance
        mass = experiment.mass

        return - (vertical_position_of_mass - z) * G * mass / sqrt((vertical_position_of_mass - z) ** 2 + r ** 2) ** 3

    else:
        raise ValueError("Test Mass Configuration Unknown.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Create new lookup table for the test masses ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_mass_parameters(
    test_mass: TestMass,
    ) -> str:

    """ All parameters that determine the potential and acceleration of the test mass: the geometry of the experiment (all attributes except its name), the radial distance from the baseline
    and the mass density. Example: 'mass=1.25_mean_radial_distance=0.073_radial_distance=0.0_mass_density=19250.0'.
    Numbers are written as floats, s.t. e.g. a radial distance of 0 and 0.0 share one lookup table. """
    parameters = {key: value for key, value in sorted(vars(test_mass.experiment).items()) if key != "name"}
    parameters["radial_distance"] = test_mass.radial_distance
    parameters["mass_density"] = test_mass.mass_density

    return "_".join(f"{key}={float(value)!r}" for key, value in parameters.items())


def test_mass_lookup_table_path(
    test_mass: TestMass,
    ) -> str:

    """ The lookup table is only valid for the parameters it was created with, hence they are part of the file name. """
    return str(LOOKUP_TABLE_DIRECTORY / ("test_mass_lookup_table_" + str(test_mass.experiment.name) + "_" + test_mass_parameters(test_mass = test_mass) + ".txt"))


def create_new_lookup_table(
        test_mass: TestMass
    ) -> None:

    # Domain of the lookup table for the tungsten block potential and acceleration
    z_min = -10         # in meters
    z_max = 10          # in meters
    n_points = 100000   # Sample points for the lookup table

    z_grid = linspace(z_min, z_max, n_points)

    print("Create lookup table for tungsten block potential and acceleration.")

    pot_table = array([
        test_mass_potential(
            z = z,
            test_mass = test_mass,
            vertical_position_of_mass = 0
        ) for z in z_grid
    ])

    print("Potential: New lookup table created.")

    acc_table = array([
        test_mass_acceleration(
            z = z,
            test_mass = test_mass,
            vertical_position_of_mass = 0
        ) for z in z_grid
    ])

    print("Acceleration: New lookup table created.")

    with open(test_mass_lookup_table_path(test_mass = test_mass), "w") as file:

        file.write("Height in [m], Potential in [J/kg], Acceleration in [m/s^2] \n")
        for height, potential, acceleration in zip(z_grid, pot_table, acc_table, strict = True):
            file.write(f"{height}, {potential}, {acceleration}\n")

    file.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Fast algorithm for test masses ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def read_lookup_table(
        test_mass: TestMass
    ) -> None:

    try:
        with open(test_mass_lookup_table_path(test_mass = test_mass)):
            pass

    except FileNotFoundError:
        create_new_lookup_table(test_mass = test_mass)


# Cache for initialized test mass lookup tables to avoid repeated file I/O
_initialize_test_mass_cache: dict[tuple, tuple] = {}

def _make_test_mass_cache_key(test_mass, potential_or_acceleration):
    """Create a hashable cache key from test_mass parameters. As the lookup table, it depends on all parameters of the test mass."""
    if test_mass is None:
        return (None, potential_or_acceleration)
    return (
        test_mass_lookup_table_path(test_mass = test_mass),
        potential_or_acceleration
    )

def initialize_test_mass(
        test_mass: TestMass,
        potential_or_acceleration: str = "Both"
    ) -> tuple:

    # Check cache first
    cache_key = _make_test_mass_cache_key(test_mass = test_mass, potential_or_acceleration = potential_or_acceleration)
    if cache_key in _initialize_test_mass_cache:
        return _initialize_test_mass_cache[cache_key]

    z_grid_list_test_mass: list[float] = []
    pot_table_list_test_mass: list[float] = []
    acc_table_list_test_mass: list[float] = []

    read_lookup_table(test_mass = test_mass)

    with open(test_mass_lookup_table_path(test_mass = test_mass)) as file:
        for line in file:
            # Strip leading and trailing whitespace, including newline characters
            line = line.strip()
            try:
                parts = line.split(',', 2)

                z_grid_list_test_mass.append(float(parts[0].strip()))
                pot_table_list_test_mass.append(float(parts[1].strip()))
                acc_table_list_test_mass.append(float(parts[2].strip()))

            except (ValueError, IndexError):
                continue

    file.close()

    # Only the requested lookup is interpolated, the other one is None.
    pot_lookup_test_mass: interp1d | None
    acc_lookup_test_mass: interp1d | None

    if potential_or_acceleration == "Potential":
        pot_lookup_test_mass = interp1d(z_grid_list_test_mass, pot_table_list_test_mass, kind = 'cubic', bounds_error = False, fill_value = "extrapolate")
        acc_lookup_test_mass = None

    elif potential_or_acceleration == "Acceleration":
        pot_lookup_test_mass = None
        acc_lookup_test_mass = interp1d(z_grid_list_test_mass, acc_table_list_test_mass, kind = 'cubic', bounds_error = False, fill_value = "extrapolate")

    else:
        pot_lookup_test_mass = interp1d(z_grid_list_test_mass, pot_table_list_test_mass, kind = 'cubic', bounds_error = False, fill_value = "extrapolate")
        acc_lookup_test_mass = interp1d(z_grid_list_test_mass, acc_table_list_test_mass, kind = 'cubic', bounds_error = False, fill_value = "extrapolate")

    result = (pot_lookup_test_mass, acc_lookup_test_mass)
    _initialize_test_mass_cache[cache_key] = result
    return result


def tungsten_potential_fast(
        z: float,
        z_center: float,
        test_mass: TestMass | None
    ) -> float:

    if test_mass is not None:
        pot_lookup_test_mass = initialize_test_mass(test_mass = test_mass, potential_or_acceleration = "Potential")[0]

        return float(pot_lookup_test_mass(z - z_center))

    else:
        return 0.0


def tungsten_acceleration_fast(
        z: float,
        z_center: float,
        test_mass: TestMass | None
    ) -> float:

    if test_mass is not None:
        acc_lookup_test_mass = initialize_test_mass(test_mass = test_mass, potential_or_acceleration = "Acceleration")[1]

        return float(acc_lookup_test_mass(z - z_center))

    else:
        return 0.0


def multiple_tungsten_blocks_potential_fast(
        z: float,
        z_centers: Sequence[float] | float,
        test_mass: TestMass | None
    ) -> float:

    if isinstance(z_centers, (int, float)):

        return tungsten_potential_fast(
            z = z,
            z_center = z_centers,
            test_mass = test_mass
        )

    else:
        total_potential = 0.
        for center in z_centers:
            total_potential += tungsten_potential_fast(
                z = z,
                z_center = float(center),
                test_mass = test_mass
            )

        return total_potential


def multiple_tungsten_blocks_acceleration_fast(
        z: float,
        z_centers: Sequence[float] | float,
        test_mass: TestMass | None
    ) -> float:

    if isinstance(z_centers, (int, float)):

        return tungsten_acceleration_fast(
            z = z,
            z_center = z_centers,
            test_mass = test_mass
        )

    else:
        total_acceleration = 0.
        for center in z_centers:
            total_acceleration += tungsten_acceleration_fast(
                z = z,
                z_center = float(center),
                test_mass = test_mass
            )

        return total_acceleration


# END OF FILE
