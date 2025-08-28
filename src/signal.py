"""
Signal processing functions for radar simulation
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, fft2, fftshift

# Try to import Rust accelerated functions
try:
    import radar_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust acceleration not available. Using pure Python implementation.")


class SignalProcessor:
    """Radar signal processing algorithms"""
    
    def __init__(self, sample_rate: float, bandwidth: float):
        """
        Initialize signal processor
        
        Args:
            sample_rate: Sampling rate in Hz
            bandwidth: Signal bandwidth in Hz
        """
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.c = 3e8  # Speed of light
        
    def matched_filter(self, received_signal: np.ndarray, 
                      reference_signal: np.ndarray) -> np.ndarray:
        """
        Apply matched filtering for pulse compression
        
        Args:
            received_signal: Received signal samples
            reference_signal: Reference pulse signal
            
        Returns:
            Matched filter output
        """
        # Use Rust implementation if available for better performance
        if RUST_AVAILABLE and len(received_signal) > 1000:
            return radar_core.matched_filter_rust(
                received_signal.astype(np.complex128),
                reference_signal.astype(np.complex128)
            )
        
        # Fallback to Python implementation
        # Matched filter is time-reversed complex conjugate of reference
        matched_filter = np.conj(reference_signal[::-1])
        
        # Apply convolution (correlation)
        output = scipy_signal.convolve(received_signal, matched_filter, mode='same')
        
        return output
    
    def pulse_compression(self, signal: np.ndarray, 
                         chirp_rate: float,
                         pulse_width: float) -> np.ndarray:
        """
        Perform pulse compression on linear FM chirp signal
        
        Args:
            signal: Received signal
            chirp_rate: Chirp rate in Hz/s
            pulse_width: Pulse width in seconds
            
        Returns:
            Compressed pulse
        """
        # Generate reference chirp
        t = np.arange(len(signal)) / self.sample_rate
        reference = self.generate_chirp(pulse_width, chirp_rate, len(signal))
        
        # Apply matched filtering
        compressed = self.matched_filter(signal, reference)
        
        return compressed
    
    def generate_chirp(self, pulse_width: float, 
                      chirp_rate: float,
                      num_samples: int) -> np.ndarray:
        """
        Generate linear frequency modulated (LFM) chirp signal
        
        Args:
            pulse_width: Pulse width in seconds
            chirp_rate: Chirp rate in Hz/s
            num_samples: Number of samples
            
        Returns:
            Complex chirp signal
        """
        t = np.linspace(0, pulse_width, num_samples)
        
        # Linear FM: phase = 2*pi*(f0*t + 0.5*k*t^2)
        # where k is chirp rate
        phase = 2 * np.pi * 0.5 * chirp_rate * t**2
        
        return np.exp(1j * phase)
    
    def cfar_detector(self, signal: np.ndarray,
                     num_guard: int = 2,
                     num_train: int = 16,
                     pfa: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cell-Averaging CFAR (CA-CFAR) detector
        
        Args:
            signal: Input signal magnitude
            num_guard: Number of guard cells on each side
            num_train: Number of training cells on each side
            pfa: Probability of false alarm
            
        Returns:
            Detection mask and threshold values
        """
        # Use Rust implementation if available for better performance
        if RUST_AVAILABLE and len(signal) > 500:
            signal_mag = np.abs(signal)
            return radar_core.cfar_detect_rust(signal_mag, num_guard, num_train, pfa)
        
        # Fallback to Python implementation
        signal_power = np.abs(signal)**2
        num_cells = len(signal)
        threshold = np.zeros(num_cells)
        detections = np.zeros(num_cells, dtype=bool)
        
        # CFAR scaling factor
        alpha = num_train * (pfa**(-1/num_train) - 1)
        
        window_size = 2 * (num_guard + num_train) + 1
        cell_under_test = num_guard + num_train
        
        for i in range(cell_under_test, num_cells - cell_under_test):
            # Left training cells
            left_train = signal_power[i - cell_under_test:i - num_guard]
            # Right training cells
            right_train = signal_power[i + num_guard + 1:i + cell_under_test + 1]
            
            # Noise power estimate
            noise_power = np.mean(np.concatenate([left_train, right_train]))
            
            # Adaptive threshold
            threshold[i] = alpha * noise_power
            
            # Detection
            if signal_power[i] > threshold[i]:
                detections[i] = True
        
        return detections, threshold
    
    def doppler_processing(self, pulses: np.ndarray, 
                          prf: float,
                          num_ffts: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Doppler processing on pulse train
        
        Args:
            pulses: 2D array of pulses (rows: range bins, cols: pulses)
            prf: Pulse repetition frequency
            num_ffts: Number of FFT points (default: number of pulses)
            
        Returns:
            Doppler spectrum and velocity bins
        """
        num_range_bins, num_pulses = pulses.shape
        
        if num_ffts is None:
            num_ffts = num_pulses
        
        # Apply window to reduce sidelobes
        window = np.hamming(num_pulses)
        windowed_pulses = pulses * window[np.newaxis, :]
        
        # Perform FFT across pulses (slow-time)
        doppler_spectrum = fft(windowed_pulses, n=num_ffts, axis=1)
        doppler_spectrum = fftshift(doppler_spectrum, axes=1)
        
        # Calculate velocity bins
        doppler_freqs = fftshift(fftfreq(num_ffts, 1/prf))
        wavelength = self.c / self.bandwidth  # Approximate
        velocities = doppler_freqs * wavelength / 2
        
        return np.abs(doppler_spectrum), velocities
    
    def range_doppler_map(self, data: np.ndarray,
                         range_resolution: float,
                         velocity_resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate range-Doppler map
        
        Args:
            data: 2D complex data (range x Doppler)
            range_resolution: Range resolution in meters
            velocity_resolution: Velocity resolution in m/s
            
        Returns:
            Range-Doppler map, range bins, velocity bins
        """
        num_range, num_doppler = data.shape
        
        # Generate range and velocity axes
        ranges = np.arange(num_range) * range_resolution
        velocities = (np.arange(num_doppler) - num_doppler // 2) * velocity_resolution
        
        # Compute power in dB
        rd_map = 20 * np.log10(np.abs(data) + 1e-10)
        
        return rd_map, ranges, velocities
    
    def mti_filter(self, pulses: np.ndarray, 
                  filter_type: str = "two_pulse") -> np.ndarray:
        """
        Moving Target Indicator (MTI) filter
        
        Args:
            pulses: Pulse data (range x pulse)
            filter_type: Type of MTI filter ('two_pulse', 'three_pulse')
            
        Returns:
            MTI filtered output
        """
        if filter_type == "two_pulse":
            # Two-pulse canceller: y[n] = x[n] - x[n-1]
            mti_output = np.diff(pulses, axis=1)
            
        elif filter_type == "three_pulse":
            # Three-pulse canceller: y[n] = x[n-2] - 2*x[n-1] + x[n]
            if pulses.shape[1] < 3:
                raise ValueError("Need at least 3 pulses for three-pulse MTI")
            
            mti_output = np.zeros((pulses.shape[0], pulses.shape[1] - 2), dtype=complex)
            mti_output = pulses[:, 2:] - 2 * pulses[:, 1:-1] + pulses[:, :-2]
            
        else:
            raise ValueError(f"Unknown MTI filter type: {filter_type}")
        
        return mti_output
    
    def coherent_integration(self, pulses: np.ndarray) -> np.ndarray:
        """
        Perform coherent integration of pulses
        
        Args:
            pulses: Complex pulse data
            
        Returns:
            Coherently integrated signal
        """
        return np.sum(pulses, axis=-1)
    
    def non_coherent_integration(self, pulses: np.ndarray) -> np.ndarray:
        """
        Perform non-coherent integration of pulses
        
        Args:
            pulses: Complex pulse data
            
        Returns:
            Non-coherently integrated signal
        """
        return np.sum(np.abs(pulses)**2, axis=-1)
    
    def estimate_noise_floor(self, signal: np.ndarray, 
                           percentile: float = 10) -> float:
        """
        Estimate noise floor from signal
        
        Args:
            signal: Signal samples
            percentile: Percentile to use for noise estimation
            
        Returns:
            Estimated noise power
        """
        signal_power = np.abs(signal)**2
        noise_estimate = np.percentile(signal_power, percentile)
        
        return noise_estimate
    
    def calculate_snr(self, signal: np.ndarray, 
                     noise_floor: Optional[float] = None) -> float:
        """
        Calculate signal-to-noise ratio
        
        Args:
            signal: Signal samples
            noise_floor: Known noise floor (if None, estimates it)
            
        Returns:
            SNR in dB
        """
        signal_power = np.max(np.abs(signal)**2)
        
        if noise_floor is None:
            noise_floor = self.estimate_noise_floor(signal)
        
        snr = 10 * np.log10(signal_power / noise_floor)
        
        return snr
    
    def ambiguity_function(self, signal: np.ndarray,
                          max_delay: int = 100,
                          max_doppler: int = 100) -> np.ndarray:
        """
        Calculate ambiguity function for waveform analysis
        
        Args:
            signal: Complex signal waveform
            max_delay: Maximum delay in samples
            max_doppler: Maximum Doppler in bins
            
        Returns:
            2D ambiguity function (delay x Doppler)
        """
        sig_len = len(signal)
        ambiguity = np.zeros((2 * max_delay + 1, 2 * max_doppler + 1), dtype=complex)
        
        for tau_idx, tau in enumerate(range(-max_delay, max_delay + 1)):
            for fd_idx, fd in enumerate(range(-max_doppler, max_doppler + 1)):
                # Doppler shift
                doppler = np.exp(1j * 2 * np.pi * fd * np.arange(sig_len) / sig_len)
                shifted_signal = signal * doppler
                
                # Time delay and correlation
                if tau >= 0:
                    if tau < sig_len:
                        ambiguity[tau_idx, fd_idx] = np.sum(
                            signal[tau:] * np.conj(shifted_signal[:sig_len - tau])
                        )
                else:
                    if -tau < sig_len:
                        ambiguity[tau_idx, fd_idx] = np.sum(
                            signal[:sig_len + tau] * np.conj(shifted_signal[-tau:])
                        )
        
        return np.abs(ambiguity)**2
    
    def clutter_suppression(self, signal: np.ndarray,
                          clutter_velocity: float = 0,
                          notch_width: float = 10) -> np.ndarray:
        """
        Suppress clutter using notch filter
        
        Args:
            signal: Input signal
            clutter_velocity: Clutter velocity in m/s
            notch_width: Notch filter width in m/s
            
        Returns:
            Clutter-suppressed signal
        """
        # Convert to frequency domain
        spectrum = fft(signal)
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        
        # Calculate clutter frequency
        wavelength = self.c / self.bandwidth
        clutter_freq = 2 * clutter_velocity / wavelength
        
        # Create notch filter
        notch_filter = np.ones_like(freqs)
        notch_mask = np.abs(freqs - clutter_freq) < notch_width / wavelength
        notch_filter[notch_mask] = 0.1  # Strong attenuation in notch
        
        # Apply filter
        filtered_spectrum = spectrum * notch_filter
        
        # Convert back to time domain
        filtered_signal = np.fft.ifft(filtered_spectrum)
        
        return filtered_signal