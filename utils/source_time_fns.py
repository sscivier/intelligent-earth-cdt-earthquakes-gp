import numpy as np


def ricker_wavelet(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
):
    """
    Generates a Ricker wavelet, also known as the "Mexican hat" wavelet.

    The Ricker wavelet is a symmetric waveform commonly used in geophysics
    to model seismic sources. It integrates to zero and has a defined peak frequency.

    Args:
        time (np.ndarray): Time vector (seconds).
        center_frequency (float): Central frequency of the wavelet (Hz).
        time_shift (float): Time at which the wavelet is centered (seconds).

    Returns:
        np.ndarray: Ricker wavelet values at each time step.
    """
    
    term = np.pi * center_frequency * (time - time_shift)
    return (1.0 - 2.0 * term ** 2) * np.exp(-term ** 2)


def gaussian_derivative(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
):
    """
    Generates a first derivative of a Gaussian function as a source time function.

    This function models the temporal derivative of a Gaussian. It integrates
    to zero and has a peak frequency determined by the central frequency.

    Args:
        time (np.ndarray): Time vector (seconds).
        center_frequency (float): Central frequency of the wavelet (Hz).
        time_shift (float): Time at which the wavelet is centered (seconds).

    Returns:
        np.ndarray: Gaussian derivative values at each time step.
    """
    
    term = np.pi * center_frequency * (time - time_shift)
    return -2.0 * term * np.exp(-term ** 2)


def sinusoidal_decay(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
    damping_factor: float = 1.0,
):
    """
    Generates a sinusoidal decay function with a given frequency and damping.

    The sinusoidal decay represents a damped harmonic oscillation that starts
    at the specified time. The damping factor controls the rate of exponential decay.

    Args:
        time (np.ndarray): Time vector (seconds).
        center_frequency (float): Central frequency of the sinusoidal component (Hz).
        time_shift (float): Time at which the decay starts (seconds).
        damping_factor (float): Damping factor controlling exponential decay rate.

    Returns:
        np.ndarray: Sinusoidal decay function values at each time step.
    """
    
    term = 2 * np.pi * center_frequency * (time - time_shift)
    decay = np.exp(-damping_factor * (time - time_shift)) * np.sin(term)
    decay[time < time_shift] = 0.0
    return decay - np.mean(decay)


def double_couple(
    time: np.ndarray,
    frequency: float,
    rise_time: float,
    duration: float,
    time_shift: float = 0.0,
):
    """
    Generates a double-couple source time function for seismic events.

    The double-couple source models the slip rate for an earthquake with a
    linear rise to peak followed by a linear fall to zero. The slip rate is
    modulated by a sinusoidal function to include oscillatory behavior.

    Args:
        time (np.ndarray): Time vector (seconds).
        frequency (float): Frequency of the oscillatory component (Hz).
        rise_time (float): Time for the slip to rise to its peak (seconds).
        duration (float): Total duration of the event (seconds).
        time_shift (float): Time at which the source starts (seconds).

    Returns:
        np.ndarray: Double-couple source function values at each time step.
    """
    
    slip_rate = np.zeros_like(time)
    for i, ti in enumerate(time):
        if time_shift <= ti <= time_shift + rise_time:
            slip_rate[i] = (ti - time_shift) / rise_time
        elif time_shift + rise_time <= ti <= time_shift + rise_time + duration:
            slip_rate[i] = 1. - (ti - time_shift - rise_time) / (duration)

    source = slip_rate * np.sin(2 * np.pi * frequency * (time - time_shift))

    return source - np.mean(source)
