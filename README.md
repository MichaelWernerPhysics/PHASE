# PHASE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24396681.svg)](https://doi.org/10.5281/zenodo.24396681)

PHASE computes the phase shifts of light-pulse atom interferometers numerically. It propagates both arms of an interferometer through the gravitational field
with 45-digit `Decimal` precision and returns the propagation, separation and laser phase, without the approximations of the usual analytic derivations.

## What PHASE can do

- **Arbitrary pulse sequences:** Mach-Zehnder, multi-diamond and other geometries, large momentum transfer (LMT, as instantaneous n-photon kicks) and relaunches.
- **Two-photon and single photon transitions:** Raman/Bragg pulses as well as clock transitions with internal states, including the mass defect of the excited state.
- **Finite speed of light:** The photon travel time to each arm is resolved for every pulse, optionally with Doppler tuned lasers.
- **Gravitational field:** Homogeneous or with a constant gradient, with g and the gravity gradient as parameters, plus additional global accelerations.
- **Signals and noise:** Gravitational waves, white laser phase noise, white mirror jitter, gravity gradient noise (underground, seismic and atmospheric models) and tungsten test masses.
- **Tides:** The tidal acceleration of the Sun and the Moon for any location and time (astropy ephemerides).

## Installation

PHASE uses [pixi](https://pixi.sh) for a reproducible environment (Python 3.12, conda-forge):

```bash
pixi install          # Creates the environment and installs PHASE in editable mode
pixi run tutorial     # Runs examples/phase_tutorial.py
pixi run test         # Runs the test suite (pixi run test -m "not slow" skips the slow tests)
```

Without pixi, `pip install -e .` installs PHASE and its dependencies into an existing environment with Python 3.12 or newer (tested up to 3.14).

## Quickstart

```python
from decimal import Decimal
import scipy.constants as scp
from phase import mach_zehnder, numerical_phases, DEFAULT_G

k_eff = Decimal(4 * scp.pi / 780e-9)            # Rb-87 Raman transition: k_eff = 2 x 2 pi / 780 nm

# Beam splitter - mirror - beam splitter with T = 1 s. Two-photon pulses need the k-vectors of both beams, k_eff = k_1 + k_2.
mzi = mach_zehnder(T = 1, k_vector = [k_eff / 2, k_eff / 2])

phases = numerical_phases(
    interferometer = mzi,
    initial_height = 0,                         # Launched upwards from z = 0 m ...
    initial_velocity = 10,                      # ... with 10 m/s
    gravity_model = "constant gradient",
)

total_phase = phases[0] + phases[1] + phases[2]  # Propagation + separation + laser phase in rad
print(f"{float(total_phase):.6e} rad, compared to -k_eff g T^2 = {float(-k_eff * DEFAULT_G):.6e} rad")
```

The tutorial [examples/phase_tutorial.py](examples/phase_tutorial.py) goes through the main features step by step and compares every result with the analytic expectation:
the Mach-Zehnder gravimeter, gravity gradient and recoil terms, the finite speed of light, single photon clock interferometers, LMT and multi-diamond geometries,
gradiometers, gravitational waves, laser phase noise, test masses, tides and gravity gradient noise. It runs in about ten seconds.

## Conventions

- **Geometry:** Heights z point upwards and are given in m. The lasers are part of the `Interferometer`: the lower laser (or the retro-reflection mirror of two-photon pulses)
  sits at `lower_laser_height` (default -1 m), the upper laser one `baseline` (default 12 m) above it. Interferometers of a differential measurement share the lasers.
  The laser phase refers to the laser that emits the light (the retro-reflection mirror for two-photon pulses), i.e. the kick phase does not depend on the choice of the origin.
- **Reference height z = 0:** g (and Γ) are given at z = 0, and so is `clock_frequency`: An atom in the excited state at the height z sees the redshifted transition frequency.
- **Gravity:** g > 0 points downwards. The model `"constant gradient"` uses g(z) = g - Γ z. The defaults are standard gravity, g = 9.80665 m/s², and the free-air gradient, Γ = 3.086e-6 1/s².
  Both can be passed to `numerical_phases` as `g` and `gravity_gradient`.
- **Interferometer:** For P pulses, the momentum lists (in units of the photon momentum) and internal state lists of both arms have P + 1 entries (before, between and after the pulses),
  `interaction_type` and `detunings` have P entries and `time_intervals` has P - 1 entries. Single photon pulses need the light direction, `"Up"` or `"Down"`, if the finite speed of light is included.
  `k_vector` has no default: the k-vector of the photon for single photon pulses (with the finite speed of light, k c has to equal `clock_frequency`), `[k_1, k_2]` for two-photon pulses.
  See [src/phase/geometries.py](src/phase/geometries.py) for examples.
- **Result:** `numerical_phases` returns `[propagation phase, separation phase, laser phase, upper arm trajectory, lower arm trajectory]`. If an atom leaves the baseline between the lasers
  before the detection, the phase has no meaning and `numerical_phases` raises an `AtomOutsideBaselineError` (a `ValueError`).
- **Accuracy:** The phases are evaluated with 45 significant digits, with `time_discretization = 1000` steps per pulse separation. The trajectories are exact in a homogeneous field and
  first order in the time step otherwise. The propagation phase integrates the Lagrangian along each step (Simpson's rule), so the trajectory error only enters to second order:
  1000 steps resolve the gravity gradient term, the finite speed of light and test mass signals to about 1e-9 (relative), also for unequal pulse separations.

## Known limitations

- **One dimension:** Atoms and light move along the vertical axis. Rotations (Sagnac and Coriolis terms), transverse motion and wavefront aberrations are not modeled.
- **Instantaneous pulses:** Every laser pulse is a momentum kick at the moment the light reaches the atom. Finite pulse durations, Rabi dynamics, populations and contrast are not modeled, PHASE returns phases only.
- **Momentum kick:** The kick velocity includes the first and second order Doppler shift and the mass defect of the excited state, but no higher order relativistic corrections.
- **Mirror jitter** only enters two-photon pulses. For single photon pulses, it is ignored.
- **Laser phase noise of two-photon pulses** is evaluated at one emission time per pulse, the one of the retro-reflected beam. The direct beam is emitted 2 (z_atom - z_mirror) / c later
  and would pick up its own laser phase noise. This delay is neglected.

Trajectories in three dimensions (with the Coriolis and Eötvös effect and wavefront aberrations) and finite pulse durations are planned, see the [roadmap](https://github.com/MichaelWernerPhysics/PHASE/wiki#roadmap).

## Package layout

| Module | Content |
| --- | --- |
| `phase.numerical_interferometer` | `numerical_phases` and the wrappers for single phase contributions, differential measurements and gradiometers |
| `phase.classes` | `Interferometer`, `GravWave`, test mass geometries, `GGN_Noise_Model`, `location` |
| `phase.geometries` | Ready-made interferometers: `mach_zehnder`, `create_MZI` (multi-diamond with LMT) |
| `phase.gravitational_model` | Gravitational potential and acceleration, mirror jitter, gravity gradient noise, tungsten test masses |
| `phase.laser_phase_noise_model` | White laser phase noise |
| `phase.earth_moon_gravity_model` | Tidal acceleration and potential of the Sun and the Moon |
| `phase.decimal_additional_functions` | `sin`, `cos`, `exp` and `pi` in `Decimal` precision |

## Authors and acknowledgment

- Michael Werner (<werner@iqo.uni-hannover.de>): main author.
- Maike Reckermann (Leibniz University Hannover): implementation of the atmospheric gravity gradient noise model.

Contributions are welcome. The [PHASE Roadmap](https://github.com/users/MichaelWernerPhysics/projects/1) shows the planned work. Please open an [issue](https://github.com/MichaelWernerPhysics/PHASE/issues) or write an e-mail.

## Citation

If you use PHASE in your research, please cite it as given in [CITATION.cff](CITATION.cff). The DOI [10.5281/zenodo.24396681](https://doi.org/10.5281/zenodo.24396681) refers to all versions of PHASE and resolves to the latest one.
To cite the exact version you used, take its DOI from the [Zenodo record](https://doi.org/10.5281/zenodo.24396681).

## References

- J. Peterson, "Observations and modeling of seismic background noise", USGS Open-File Report 93-322 (1993): New Low/High Noise Models.
- A. Peters, K. Y. Chung and S. Chu, "High-precision gravity measurements using atom interferometry", Metrologia 38, 25 (2001): gravity gradient terms.
- P. W. Graham, J. M. Hogan, M. A. Kasevich and S. Rajendran, "New method for gravitational wave detection with atomic sensors", Phys. Rev. Lett. 110, 171102 (2013): single photon gradiometers.
- L. Badurina, V. Gibson, C. McCabe and J. Mitchell, "Ultralight dark matter searches at the sub-Hz frontier with atom multigradiometry", Phys. Rev. D 107, 055002 (2023): seismic gravity gradient noise underground.
- M. Reckermann et al., "Dual-species atom interferometry for axion dark matter", arXiv:2611.15874 (2026): seismic gravity gradient noise above ground and atmospheric gravity gradient noise (Appendix A.3).
- J. Carlton, V. Gibson, T. Kovachy, C. McCabe and J. Mitchell, "Characterizing atmospheric gravity gradient noise for vertical atom interferometers", Phys. Rev. D 111, 082003 (2025): atmospheric gravity gradient noise.
- J. R. Bowman, G. E. Baker and M. Bahavar, "Ambient infrasound noise", Geophys. Res. Lett. 32 (2005): infrasound pressure spectra.
- S. K. Kristoffersen, A. Le Pichon, P. Hupe and R. S. Matoza, "Updated global reference models of broadband coherent infrasound signals for atmospheric studies and civilian applications", Earth Space Sci. 9, e2022EA002222 (2022).

## License

PHASE is released under the [MIT License](LICENSE).
