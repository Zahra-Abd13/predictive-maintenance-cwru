import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq




def time_domain_features(signal_segment: np.ndarray) -> dict:
    """
    Extract 9 statistical features from the time-domain signal.

    These features capture the overall shape and energy of the
    vibration waveform without any frequency transformation.

    Features
    --------
    rms          : Root Mean Square — overall vibration energy level
    kurtosis     : 4th statistical moment — spikes from bearing impacts
                   (healthy ≈ 3, faulty can reach 10–30)
    crest_factor : peak / RMS — sharpness of impulse peaks
    skewness     : asymmetry of the amplitude distribution
    peak         : maximum absolute amplitude
    peak_to_peak : total swing from min to max
    std          : standard deviation — spread of amplitudes
    mean_abs     : mean of absolute values — average rectified level
    shape_factor : RMS / mean_abs — waveform shape index

    Parameters
    ----------
    window : 1-D numpy array — one signal segment

    Returns
    -------
    dict of feature_name -> float
    """
    eps = 1e-10

    rms          = np.sqrt(np.mean(signal_segment ** 2))
    kurt         = kurtosis(signal_segment)          
    skewness     = skew(signal_segment)
    peak         = np.max(np.abs(signal_segment))
    crest_factor = peak / (rms + eps)
    p2p          = np.max(signal_segment) - np.min(signal_segment)
    std          = np.std(signal_segment)
    mean_abs     = np.mean(np.abs(signal_segment))
    shape_factor = rms / (mean_abs + eps)

    return {
        'rms':          rms,
        'kurtosis':     kurt,
        'crest_factor': crest_factor,
        'skewness':     skewness,
        'peak':         peak,
        'peak_to_peak': p2p,
        'std':          std,
        'mean_abs':     mean_abs,
        'shape_factor': shape_factor,
    }




def frequency_domain_features(signal_segment: np.ndarray,
                               fs: float = 12000.0) -> dict:
    """
    Extract 7 features from the frequency-domain (FFT) of the signal.

    Frequency-domain features reveal which frequency components are
    present — bearing faults produce energy at specific characteristic
    frequencies (BPFI, BPFO, BSF) that are invisible in time domain.

    Features
    --------
    dominant_freq          : frequency (Hz) with highest FFT magnitude
    spectral_energy        : total energy in the frequency spectrum
    spectral_entropy       : randomness/complexity of the spectrum
    spectral_mean          : mean FFT magnitude
    spectral_std           : std FFT magnitude
    band_energy_100_4000   : energy ratio in 100–4000 Hz
    band_energy_4000_6000  : energy ratio in 4000–6000 Hz

    Parameters
    ----------
    window : 1-D numpy array — one signal segment
    fs     : sampling rate in Hz (CWRU default = 12000)

    Returns
    -------
    dict of feature_name -> float
    """

    eps = 1e-10
    N = len(signal_segment)

    # ──────────────────────────────────────────
    # Apply Hanning window before FFT
    # Reduces spectral leakage
    # ──────────────────────────────────────────
    hanning_fn = np.hanning(N)
    windowed_signal = signal_segment * hanning_fn

    # FFT
    fft_magnitudes = np.abs(rfft(windowed_signal))
    freqs = rfftfreq(N, d=1.0 / fs)

    # Total spectral energy
    total_energy = np.sum(fft_magnitudes ** 2) + eps

    # Dominant frequency
    dominant_freq = freqs[np.argmax(fft_magnitudes)]

    # Spectral entropy
    psd = fft_magnitudes ** 2 / total_energy
    spectral_entropy = -np.sum(psd * np.log(psd + eps))

    # Frequency bands
    mask_fault = (freqs >= 100) & (freqs <= 4000)
    mask_noise = (freqs > 4000) & (freqs <= 6000)

    band_energy_fault = (
        np.sum(fft_magnitudes[mask_fault] ** 2)
        / total_energy
    )

    band_energy_noise = (
        np.sum(fft_magnitudes[mask_noise] ** 2)
        / total_energy
    )

    return {
        'dominant_freq': dominant_freq,
        'spectral_energy': total_energy,
        'spectral_entropy': spectral_entropy,
        'spectral_mean': np.mean(fft_magnitudes),
        'spectral_std': np.std(fft_magnitudes),
        'band_energy_100_4000': band_energy_fault,
        'band_energy_4000_6000': band_energy_noise,
    }




def extract_features(signal_segment: np.ndarray,
                     fs: float = 12000.0) -> dict:
    """
    Extract all 16 features (time + frequency) from one signal window.

    This is the main function you call in notebook 03_features.ipynb.

    Parameters
    ----------
    window : 1-D numpy array — one preprocessed signal segment
    fs     : sampling rate in Hz

    Returns
    -------
    dict with 16 feature keys and their float values
    """
    time_feats = time_domain_features(signal_segment)
    freq_feats = frequency_domain_features(signal_segment, fs=fs)

    
    return {**time_feats, **freq_feats}




def get_feature_names() -> list:
    """Return the list of all 16 feature names (without label)."""
    return [
       
        'rms', 'kurtosis', 'crest_factor', 'skewness', 'peak',
        'peak_to_peak', 'std', 'mean_abs', 'shape_factor',
       
        'dominant_freq', 'spectral_energy', 'spectral_entropy',
        'spectral_mean', 'spectral_std',
        'band_energy_100_4000', 'band_energy_4000_6000',
    ]
