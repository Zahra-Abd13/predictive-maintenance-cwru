import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt


def butterworth_lowpass(signal: np.ndarray,
                        cutoff_hz: float = 5000.0,
                        fs: float = 12000.0,
                        order: int = 5) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter.

    Removes high-frequency noise above cutoff_hz while preserving
    fault-characteristic frequencies (BPFI, BPFO, BSF).
    Uses filtfilt() for zero phase distortion — the filtered signal
    is not shifted in time relative to the original.

    Parameters
    ----------
    signal     : 1-D numpy array — raw vibration signal
    cutoff_hz  : frequency above which to remove noise (default 5000 Hz)
    fs         : sampling rate in Hz (CWRU default = 12000 Hz)
    order      : filter order — higher = sharper cutoff (default 5)

    Returns
    -------
    np.ndarray : filtered signal, same length as input
    """
    nyq = fs / 2.0
    normal_cutoff = cutoff_hz / nyq

    if normal_cutoff >= 1.0:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be less than Nyquist "
            f"frequency ({nyq}). Lower the cutoff."
        )

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


def butterworth_bandpass(signal: np.ndarray,
                         low_hz: float = 100.0,
                         high_hz: float = 5000.0,
                         fs: float = 12000.0,
                         order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Useful for isolating the fault-frequency band (100–5000 Hz)
    and removing both DC drift (very low freq) and high-freq noise.

    Parameters
    ----------
    signal  : 1-D numpy array
    low_hz  : lower cutoff frequency in Hz
    high_hz : upper cutoff frequency in Hz
    fs      : sampling rate in Hz
    order   : filter order

    Returns
    -------
    np.ndarray : band-pass filtered signal
    """
    nyq = fs / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq

    if high >= 1.0:
        raise ValueError(
            f"high_hz ({high_hz}) must be less than Nyquist ({nyq})."
        )

    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Remove the DC component (mean offset) from a signal.
    Simple but important — DC bias can distort RMS and kurtosis.

    Returns
    -------
    np.ndarray : zero-mean signal
    """
    return signal - np.mean(signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to the range [-1, 1] using min-max scaling.

    Returns
    -------
    np.ndarray : normalized signal
    """
    s_min = np.min(signal)
    s_max = np.max(signal)
    denom = s_max - s_min

    if denom == 0:
        return np.zeros_like(signal)

    return 2.0 * (signal - s_min) / denom - 1.0


def full_preprocessing_pipeline(signal: np.ndarray,
                                 fs: float = 12000.0) -> np.ndarray:
    """
    Apply the full preprocessing pipeline in order:
        1. Remove DC offset
        2. Butterworth low-pass filter (remove noise above 5 kHz)

    Note: Signal normalization is intentionally excluded.
    Amplitude energy is a discriminative feature between fault classes
    (Normal RMS ≈ 0.066 vs Outer Race RMS ≈ 0.59 — a 9× difference).
    Normalization would erase this information.
    StandardScaler is applied on extracted features in notebook 03.
    """
    signal = remove_dc_offset(signal)
    signal = butterworth_lowpass(signal, cutoff_hz=5000.0, fs=fs, order=5)
    return signal   # ← no normalize_signal


def segment_signal(signal: np.ndarray,
                   window_size: int = 1024,
                   overlap: float = 0.5) -> np.ndarray:
    """
    Split a continuous signal into overlapping windows.
    Each window becomes one training sample.

    Parameters
    ----------
    signal      : 1-D numpy array — preprocessed vibration signal
    window_size : number of samples per window (default 1024)
                  At 12 kHz, 1024 samples ≈ 85 milliseconds
    overlap     : fraction of overlap between consecutive windows
                  0.5 = 50% overlap (default)

    Returns
    -------
    np.ndarray of shape (n_windows, window_size)
    """
    step = int(window_size * (1 - overlap))
    windows = []

    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start: start + window_size]
        windows.append(window)

    return np.array(windows)




def plot_filter_comparison(raw: np.ndarray,
                           filtered: np.ndarray,
                           fs: float = 12000.0,
                           title: str = 'Raw vs Filtered Signal',
                           n_samples: int = 1024) -> None:
    """
    Plot raw and filtered signals side by side for visual comparison.
    Call this in your EDA notebook to show the denoising effect.
    """
    t = np.arange(n_samples) / fs * 1000  

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    axes[0].plot(t, raw[:n_samples], color='#ff6b6b', linewidth=0.8, alpha=0.9)
    axes[0].set_title('Raw Signal', fontsize=11)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, filtered[:n_samples], color='#4ecdc4', linewidth=0.8, alpha=0.9)
    axes[1].set_title('After Butterworth Low-pass Filter (5 kHz cutoff)', fontsize=11)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_power_spectrum(signal: np.ndarray,
                        fs: float = 12000.0,
                        title: str = 'Power Spectral Density') -> None:
    """
    Plot the Power Spectral Density (PSD) of a signal using Welch's method.
    Useful for identifying dominant fault frequencies in the frequency domain.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=1024)

    plt.figure(figsize=(12, 4))
    plt.semilogy(freqs, psd, color='#6c8cff', linewidth=0.9)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
