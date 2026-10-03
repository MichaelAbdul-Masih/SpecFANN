import numpy as np


def calc_phase(period, t0, t):
    """
    Calculate the phase of an observation given the period, reference time, and observation time.

    Parameters:
    period (array-like): The period of the orbit in days.
    t0 (array-like): The reference time in days.
    t (array-like): The observation times in days.

    Returns:
    phase (array-like): The calculated phases corresponding to the observation times.
    """
    period = np.asarray(period)
    t0 = np.asarray(t0)
    t = np.asarray(t)
    phase = ((t[None, :] - t0[:, None]) / period[:,None]) % 1
    return phase


def solve_Keplers_equation(E, phis, ecc):
    """
    Solve Kepler's equation for the eccentric anomaly.

    Parameters:
    E (array-like): The initial guess for the eccentric anomalies.
    phis (array-like): The phases.
    ecc (array-like): The eccentricities.

    Returns:
    E (array-like): The solved eccentric anomalies.
    """

    E2 = (2*np.pi*phis - ecc[:, None]*(E*np.cos(E) - np.sin(E))) / (1. - ecc[:, None]*np.cos(E))
    eps = np.abs(E2 - E)
    if np.all(eps < 1E-6):
        return E2
    else:
        return solve_Keplers_equation(E2, phis, ecc)


def calc_true_anomaly(phase, ecc):
    """
    Calculate the true anomaly from the phase and eccentricity.

    Parameters:
    phase (array-like): The phases.
    ecc (array-like): The eccentricities.

    Returns:
    true_anomaly (array-like): The calculated true anomalies.
    """
    ecc[ecc < 0] = 0
    ecc[ecc > 0.99999] = 0
    E = 2*np.pi*phase
    E = solve_Keplers_equation(E, phase, ecc)
    true_anomaly = 2*np.arctan(np.sqrt((1 + ecc[:, None])/(1 - ecc[:, None])) * np.tan(E/2))
    return true_anomaly


def calc_RVs(period, t0, ecc, omega, K, gamma, t):
    """
    Calculate the radial velocities based on the orbital parameters and observation times.

    Parameters:
    period (array-like): The periods of the orbits in days.
    t0 (array-like): The reference times in days.
    ecc (array-like): The eccentricities.
    omega (array-like): The arguments of periastron in radians.
    K (array-like): The semi-amplitudes in km/s.
    gamma (array-like): The systemic velocities in km/s.
    t (array-like): The observation times in days.

    Returns:
    RVs (array-like): The calculated radial velocities corresponding to the observation times.
    """

    phase = calc_phase(period, t0, t)
    true_anomaly = calc_true_anomaly(phase, ecc)
    RVs = K[:, None] * (np.cos(true_anomaly + omega[:, None]) + ecc[:, None]*np.cos(omega[:, None])) + gamma[:, None]
    if np.isnan(RVs).any():
        print(period, t0, ecc, omega, K, gamma, t)
    return RVs