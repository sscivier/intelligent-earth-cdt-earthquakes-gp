import numpy as np


def ricker_wavelet(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
):
    
    term = np.pi * center_frequency * (time - time_shift)
    return (1.0 - 2.0 * term ** 2) * np.exp(-term ** 2)


def gaussian_derivative(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
):
    
    term = np.pi * center_frequency * (time - time_shift)
    return -2.0 * term * np.exp(-term ** 2)


def sinusoidal_decay(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
    damping_factor: float = 1.0,
):
    
    term = np.pi * center_frequency * (time - time_shift)
    decay = np.exp(-damping_factor * (time - time_shift)) * np.sin(term)
    decay[time < time_shift] = 0.0
    return decay


def double_couple(
    time: np.ndarray,
    rise_time: float,
    duration: float,
    time_shift: float = 0.0,
):
    
    slip_rate = np.zeros_like(time)
    for i, ti in enumerate(time):
        if time_shift <= ti <= time_shift + rise_time:
            slip_rate[i] = (ti - time_shift) / rise_time
        elif time_shift + rise_time <= ti <= time_shift + rise_time + duration:
            slip_rate[i] = 1. - (ti - time_shift - rise_time) / (duration - rise_time)

    return slip_rate * np.sin(2 * np.pi * (time - time_shift) / duration)
