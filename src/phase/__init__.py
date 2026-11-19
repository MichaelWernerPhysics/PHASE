""" PHASE: Numerical phase shifts of light-pulse atom interferometers.

The main entry point is numerical_phases, which propagates both arms of an interferometer through the gravitational field and returns the propagation,
separation and laser (kick) phase. The interferometer geometry is described by the Interferometer class, see geometries for ready-made examples. """

# Single source of the version: pyproject.toml reads it from here, CITATION.cff has to match (see tests/test_package.py).
__version__ = "0.1.0"

from phase.classes import (
    GGN_Noise_Model as GGN_Noise_Model,
    GravWave as GravWave,
    Interferometer as Interferometer,
    StanfordMass as StanfordMass,
    StanfordMassSimplified as StanfordMassSimplified,
    TestMass as TestMass,
    zeroGW as zeroGW,
)
from phase.geometries import (
    create_MZI as create_MZI,
    mach_zehnder as mach_zehnder,
)
from phase.gravitational_model import (
    DEFAULT_G as DEFAULT_G,
    DEFAULT_GRAVITY_GRADIENT as DEFAULT_GRAVITY_GRADIENT,
    set_white_mirror_jitter as set_white_mirror_jitter,
)
from phase.laser_phase_noise_model import set_white_phase_noise as set_white_phase_noise
from phase.numerical_interferometer import (
    AtomOutsideBaselineError as AtomOutsideBaselineError,
    num_phase_with_origin as num_phase_with_origin,
    numerical_differential_interferometer as numerical_differential_interferometer,
    numerical_phases as numerical_phases,
    numerical_same_interferometer_height_difference as numerical_same_interferometer_height_difference,
)
