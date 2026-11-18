from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm  # type: ignore[import-untyped]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Gravity Gradient Noise ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" Gravity gradient noise (GGN) of the ground and the atmosphere in three variants, see gravity_gradient_noise_potential:
- "underground": Seismic GGN for an interferometer in a vertical shaft, i.e. underneath the surface (Badurina et al. 2023).
- "seismic": Seismic GGN for an interferometer on top of the ground (Reckermann et al. 2026, Appendix A.3.1).
- "atmospheric": Infrasound GGN of the atmosphere (Carlton et al. 2025, Reckermann et al. 2026, Appendix A.3.2), implemented by Maike Reckermann (LUH).
The seismic variants are driven by the Peterson New Low/High Noise Models (NLNM/NHNM) from obspy, the atmospheric variant by the ambient infrasound pressure spectra of Bowman et al.
(bowman_low_noise_pressure.csv and bowman_high_noise_pressure.csv).

References:
- J. Peterson, "Observations and modeling of seismic background noise", USGS Open-File Report 93-322 (1993).
- L. Badurina, V. Gibson, C. McCabe and J. Mitchell, "Ultralight dark matter searches at the sub-Hz frontier with atom multigradiometry", Phys. Rev. D 107, 055002 (2023).
- M. Reckermann et al., "Dual-species atom interferometry for axion dark matter", arXiv:2611.15874 (2026).
- J. Carlton, V. Gibson, T. Kovachy, C. McCabe and J. Mitchell, "Characterizing atmospheric gravity gradient noise for vertical atom interferometers", Phys. Rev. D 111, 082003 (2025).
- J. R. Bowman, G. E. Baker and M. Bahavar, "Ambient infrasound noise", Geophys. Res. Lett. 32 (2005), doi:10.1029/2005GL022486.
- S. K. Kristoffersen, A. Le Pichon, P. Hupe and R. S. Matoza, "Updated global reference models of broadband coherent infrasound signals for atmospheric studies
  and civilian applications", Earth Space Sci. 9, e2022EA002222 (2022). """

# The pressure spectra are stored next to this file.
DATA_DIRECTORY = Path(__file__).resolve().parent

PARAMS = dict(
    rho0 = 1800.0,  # kg/m^3
    s = 0.36,  # dimensionless
    q = 0.88,  # dimensionless
    cH = 205.0,  # m/s
    G = 6.6743e-11,  # m^3/(kg s^2)
)


def _peterson_asd(type = "NLNM"):
    """
    Return (f_grid [Hz], displacement ASD [m/sqrt(Hz)]) for NLNM/NHNM.
    """
    if type.upper() == "NLNM":
        per_s, db = get_nlnm()
    elif type.upper() == "NHNM":
        per_s, db = get_nhnm()
    else:
        raise ValueError("type must be 'NLNM' or 'NHNM'.")

    per_s = np.asarray(per_s, float)
    db = np.asarray(db, float)
    f = 1.0 / per_s  # Hz
    psd_acc = 10.0 ** (db / 10.0)  # (m^2/s^4)/Hz
    asd_disp = np.sqrt(psd_acc / ((2 * np.pi * f) ** 4))  # m/sqrt(Hz)

    idx = np.argsort(f)
    return f[idx], asd_disp[idx]


def _bowman_psd(type = "low"):

    '''
    Ambient infrasound pressure noise of Bowman et al. (2005) in Pa^2/Hz, low or high noise level.
    '''
    if type == 'low':
        f_bowman, psd_bowman = load_csv_data(filepath = DATA_DIRECTORY / "bowman_low_noise_pressure.csv")
    elif type == 'high':
        f_bowman, psd_bowman = load_csv_data(filepath = DATA_DIRECTORY / "bowman_high_noise_pressure.csv")
    else:
        raise ValueError("type must be 'high' or 'low'.")

    return f_bowman, psd_bowman


def _interp_asd(f_query, type = "NLNM", model = "seismic"):
    """
    Log-log interpolate the displacement ASD at frequencies f_query (Hz).
    Outside the Peterson band, returns 0.
    """

    if model.lower() == "atmospheric":
        f_grid, asd_grid = _bowman_psd(type = type)
        f_query = np.asarray(f_query, float)
    else:
        f_grid, asd_grid = _peterson_asd(type = type)
        f_query = np.asarray(f_query, float)

    # Clamp to grid for interpolation; zero outside
    fmin, fmax = f_grid.min(), f_grid.max()
    mask_out = (f_query < fmin) | (f_query > fmax)
    fq = np.clip(f_query, fmin, fmax)

    asd = np.exp(np.interp(np.log(fq), np.log(f_grid), np.log(asd_grid)))
    asd = np.where(mask_out, 0.0, asd)
    return asd


def _local_bandwidth(f):
    """
    Estimate a local bandwidth df per frequency sample (half-distance to neighbors).
    """
    f = np.asarray(f, float)
    if f.size == 1:
        return np.array([max(f[0], 1e-9)], dtype = float)

    idx = np.argsort(f)
    fs = f[idx]
    df_s = np.empty_like(fs)
    df_s[0] = 0.5 * (fs[1] - fs[0])
    df_s[-1] = 0.5 * (fs[-1] - fs[-2])
    df_s[1:-1] = 0.5 * (fs[2:] - fs[:-2])
    df_s = np.maximum(df_s, 1e-12)

    inv = np.empty_like(idx)
    inv[idx] = np.arange(f.size)
    return df_s[inv]


def make_modes(
    f, type = "NLNM", df = None, phi = None, rng = None, model = "seismic"
):
    """
    Build modal tuples (f [Hz], xi [m], phi [rad]) from NLNM/NHNM ASD.
    - f: array-like of modal frequencies [Hz]
    - df: scalar or array-like bandwidth per line [Hz] (if None, estimated locally)
    - phi: phases in [0, 2π). If None, drawn uniformly with rng.
    """
    f = np.asarray(f, float)
    if df is None:
        df = _local_bandwidth(f = f)
    else:
        df = np.asarray(df, float) + 0.0

    if df.shape != f.shape and df.size != 1:
        raise ValueError("df must be scalar or same shape as f.")

    df = np.broadcast_to(df, f.shape)
    df = np.maximum(df, 1e-12)

    asd = _interp_asd(f_query = f, type = type, model = model)  # m/sqrt(Hz) for Seismic GGN and PSD in Pa^2/Hz for Atmospheric GGN
    if model.lower() == "atmospheric":
        xi = np.sqrt(asd * 10 ** -10) * np.sqrt(df)   # line amplitude (1-sigma) in bar
    else:
        xi = asd * np.sqrt(df)  # line amplitude (1-sigma) in meters

    if phi is None:

        try:
            rng = np.random.default_rng(rng)
            phi = rng.uniform(0.0, 2.0 * np.pi, size = f.shape)
        except TypeError:
            phi = np.random.uniform(0.0, 2.0 * np.pi, size = f.shape)

    else:
        phi = np.asarray(phi, float)
        if phi.shape != f.shape:
            raise ValueError("phi must have same shape as f.")

    return f, xi, phi


def gravity_gradient_noise_potential(
    z: Decimal | float,
    t: Decimal | float,
    f: np.ndarray,
    xi: np.ndarray,
    phi: np.ndarray,
    params: dict[str, float] = PARAMS,
    z_offset: float = 10.0,
    model: str = "underground"
) -> float:
    """
    Stevenson GGN potential (scalar or arrays).
    Inputs:
      - z: float or array [m] (depth or height)
      - t: float or array [s] (time)
      - f, xi, phi: arrays of equal shape
    Returns:
      - V(z,t): same broadcasted shape as z and t
    """
    rho0, s, q, cH, G = (params[k] for k in ("rho0", "s", "q", "cH", "G"))

    f = np.asarray(f, float)
    xi = np.asarray(xi, float)
    phi = np.asarray(phi, float)
    if not (f.shape == xi.shape == phi.shape):
        raise ValueError("f, xi, phi must have identical shapes.")

    omega = 2.0 * np.pi * f  # rad/s
    k = omega / cH  # 1/m

    # Ignore zero-frequency modes entirely before evaluating the 1/omega factor.
    mask = omega > 0.0
    if not np.any(mask):
        return 0.0

    f = f[mask]
    xi = xi[mask]
    phi = phi[mask]
    omega = omega[mask]
    k = k[mask]

    geom = 1 / omega

    # Broadcast over modes
    omega_b = omega[np.newaxis, ...]
    k_b = k[np.newaxis, ...]
    xi_b = xi[np.newaxis, ...]
    phi_b = phi[np.newaxis, ...]
    geom_b = geom[np.newaxis, ...]

    if model.lower() == "underground":
        """ Seismic GGN for an interferometer underneath the surface, see Badurina et al. (2023). """
        z_array = np.asarray(Decimal(z_offset) - Decimal(z), float)
        t_array = np.asarray(t, float)

        # Expand z and t to broadcast with modes
        z_b = np.expand_dims(z_array, axis = -1)
        t_b = np.expand_dims(t_array, axis = -1)

        phase = omega_b * t_b + phi_b
        prefac = -2.0 * np.pi * G * rho0 * cH * ((1.0 + s**2) / (1.0 - s**2)) / q

        depth_term = (1.0 + np.sqrt(q / s)) * np.exp(-k_b * z_b) - 2.0 * np.exp(-q * k_b * z_b)
        term = (xi_b * geom_b) * np.cos(phase) * depth_term

        V = prefac * np.sum(term, axis = -1)
        return float(np.squeeze(V))

    elif model.lower() == "seismic":
        """Seismic GGN model for atom interferometers, which are on top of the ground and not underneath.  """
        z_array = np.asarray(z, float)
        t_array = np.asarray(t, float)

        # Expand z and t to broadcast with modes
        z_b = np.expand_dims(z_array, axis = -1)
        t_b = np.expand_dims(t_array, axis = -1)

        phase = omega_b * t_b + phi_b
        prefac = G * rho0 * cH * (1.0 + (q ** 2 - 1) / (q ** 2 + q) * (1.0 + s**2) / (1.0 - s**2))

        depth_term = np.exp(-omega_b * z_b / cH)
        term = (xi_b * geom_b) * np.cos(phase) * depth_term

        V = prefac * np.sum(term, axis = -1)
        return float(np.squeeze(V))

    elif model.lower() == "atmospheric":
        """Atmospheric (infrasound) GGN, see Carlton et al. (2025) and Reckermann et al. (2026), Appendix A.3.2. Implementation by Maike Reckermann (LUH)."""
        gamma_p = 1.4
        p0 = 1.013 # bar
        rho0 = 1.3  # kg/m^3, density of air (PARAMS holds the density of the ground)
        alpha = G * rho0 / (gamma_p * p0)
        prefac = 4 * np.pi * alpha

        z_array = np.asarray(z, float)
        t_array = np.asarray(t, float)

        cs = 343 # m/s
        k = omega / cs
        theta = 7/16 * np.pi

        kr = k * np.cos(theta)
        kz = k * np.sin(theta)

        z_b = np.expand_dims(z_array, axis = -1)
        t_b = np.expand_dims(t_array, axis = -1)

        phase = omega_b * t_b + phi_b
        depth_term = (np.exp(- kr * z_b) - 2 * np.cos(kz * z_b)) / k ** 2
        term = xi_b * np.cos(phase) * depth_term

        V = prefac * np.sum(term, axis = -1)
        return float(np.squeeze(V))


    raise ValueError("Model must be 'underground', 'seismic' or 'atmospheric'.")


def load_csv_data(filepath: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load the frequency and PSD columns from a CSV file without header."""
    data = np.loadtxt(filepath, delimiter = ",", ndmin = 2)
    return data[:, 0], data[:, 1]


def plot_peterson_asd(fmin = None, fmax = None, ax = None, show = False):
    """
    Plot NLNM/NHNM displacement ASDs (m/sqrt(Hz)) vs frequency (Hz). Returns the axes, e.g. to save the figure with ax.figure.savefig(...).
    """
    f_nlnm, asd_nlnm = _peterson_asd(type = "NLNM")
    f_nhnm, asd_nhnm = _peterson_asd(type = "NHNM")

    if fmin is not None or fmax is not None:
        fmin = fmin if fmin is not None else min(f_nlnm.min(), f_nhnm.min())
        fmax = fmax if fmax is not None else max(f_nlnm.max(), f_nhnm.max())
        m1 = (f_nlnm >= fmin) & (f_nlnm <= fmax)
        m2 = (f_nhnm >= fmin) & (f_nhnm <= fmax)
    else:
        m1 = slice(None)
        m2 = slice(None)

    if ax is None:
        _, ax = mpl.subplots()

    ax.loglog(f_nlnm[m1], asd_nlnm[m1], label = "NLNM")
    ax.loglog(f_nhnm[m2], asd_nhnm[m2], label = "NHNM")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Displacement ASD [m/√Hz]")
    ax.legend()
    ax.grid(True, which = "both", linestyle = ":")

    if show:
        mpl.show()
    return ax


# END OF FILE
