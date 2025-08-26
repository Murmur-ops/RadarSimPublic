"""
Enhanced waveform generation for radar simulation with realistic hardware impairments
and advanced signal processing capabilities.

This module extends the base WaveformGenerator with:
- Hardware impairments (phase noise, amplitude errors)
- Pulse train generation with proper timing
- Matched filter references
- Transmit beamforming for phased arrays
- Power amplifier effects (compression, harmonics)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
import scipy.signal as signal
from scipy.interpolate import interp1d
from enum import Enum

# Import base classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveforms import WaveformGenerator, WaveformParameters, WaveformType
from radar import RadarParameters


class BeamformingType(Enum):
    """Beamforming algorithm types"""
    CONVENTIONAL = "conventional"
    TAYLOR = "taylor"
    CHEBYSHEV = "chebyshev"
    ADAPTIVE = "adaptive"


@dataclass
class HardwareImpairments:
    """Hardware impairment parameters"""
    phase_noise_level: float = -80.0  # dBc/Hz at 1 Hz offset
    phase_noise_slope: float = -20.0  # dB/decade
    amplitude_ripple: float = 0.1  # Peak-to-peak amplitude variation (fraction)
    amplitude_drift: float = 0.05  # Long-term amplitude drift (fraction)
    timing_jitter: float = 1e-12  # RMS timing jitter (seconds)
    iq_imbalance_gain: float = 0.02  # I/Q gain imbalance (fraction)
    iq_imbalance_phase: float = 0.5  # I/Q phase imbalance (degrees)
    dc_offset_i: float = 0.001  # DC offset on I channel (fraction)
    dc_offset_q: float = 0.001  # DC offset on Q channel (fraction)


@dataclass
class PowerAmplifierModel:
    """Power amplifier characteristics"""
    max_power: float = 1000.0  # Maximum output power (Watts)
    p1db_compression: float = -1.0  # 1dB compression point (dB)
    saturation_power: float = 1200.0  # Saturation power (Watts)
    gain: float = 30.0  # Small signal gain (dB)
    phase_shift_coefficient: float = 1.0  # AM-PM conversion (deg/dB)
    harmonic_levels: Dict[int, float] = field(default_factory=lambda: {
        2: -20.0,  # 2nd harmonic level (dBc)
        3: -30.0,  # 3rd harmonic level (dBc)
        4: -40.0,  # 4th harmonic level (dBc)
    })
    thermal_time_constant: float = 0.1  # Thermal time constant (seconds)


@dataclass
class ArrayGeometry:
    """Phased array antenna geometry"""
    num_elements: int = 64
    element_spacing: float = 0.5  # Wavelengths
    array_type: str = "linear"  # 'linear', 'planar', 'circular'
    element_positions: Optional[np.ndarray] = None  # Custom positions
    element_weights: Optional[np.ndarray] = None  # Element weightings
    element_phases: Optional[np.ndarray] = None  # Element phases


class EnhancedWaveformGenerator(WaveformGenerator):
    """Enhanced waveform generator with hardware impairments and advanced features"""
    
    def __init__(self, 
                 params: WaveformParameters,
                 radar_params: Optional[RadarParameters] = None,
                 hardware_impairments: Optional[HardwareImpairments] = None,
                 pa_model: Optional[PowerAmplifierModel] = None,
                 array_geometry: Optional[ArrayGeometry] = None):
        """
        Initialize enhanced waveform generator
        
        Args:
            params: Basic waveform parameters
            radar_params: Radar system parameters
            hardware_impairments: Hardware impairment model
            pa_model: Power amplifier model
            array_geometry: Array antenna geometry
        """
        super().__init__(params)
        self.radar_params = radar_params
        self.hardware_impairments = hardware_impairments or HardwareImpairments()
        self.pa_model = pa_model or PowerAmplifierModel()
        self.array_geometry = array_geometry or ArrayGeometry()
        
        # Internal state for thermal effects
        self._thermal_state = 0.0
        self._previous_power = 0.0
        
        # Cache for matched filter references
        self._matched_filter_cache = {}
        
    def generate_pulse_train(self,
                           waveform_type: WaveformType,
                           num_pulses: int,
                           pri: float,
                           pulse_to_pulse_variation: bool = True,
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coherent pulse train with realistic timing and variations
        
        Args:
            waveform_type: Type of waveform for each pulse
            num_pulses: Number of pulses in train
            pri: Pulse Repetition Interval (seconds)
            pulse_to_pulse_variation: Enable pulse-to-pulse variations
            **kwargs: Additional waveform-specific parameters
            
        Returns:
            Tuple of (pulse_train, time_vector)
        """
        # Calculate total time and samples
        total_time = num_pulses * pri
        samples_per_pri = int(pri * self.params.sample_rate)
        pulse_samples = self.params.num_samples
        
        # Initialize pulse train
        total_samples = samples_per_pri * num_pulses
        pulse_train = np.zeros(total_samples, dtype=complex)
        time_vector = np.arange(total_samples) / self.params.sample_rate
        
        # Generate each pulse with variations
        for pulse_idx in range(num_pulses):
            # Base waveform
            pulse = self.generate(waveform_type, **kwargs)
            
            # Apply pulse-to-pulse variations if enabled
            if pulse_to_pulse_variation:
                pulse = self._apply_pulse_variations(pulse, pulse_idx)
            
            # Apply hardware impairments
            pulse = self._apply_hardware_impairments(pulse)
            
            # Apply power amplifier effects
            pulse = self._apply_power_amplifier(pulse)
            
            # Place pulse in train with timing jitter
            start_idx = pulse_idx * samples_per_pri
            timing_jitter = np.random.normal(0, self.hardware_impairments.timing_jitter)
            jitter_samples = int(timing_jitter * self.params.sample_rate)
            actual_start = max(0, min(start_idx + jitter_samples, total_samples - pulse_samples))
            
            end_idx = min(actual_start + pulse_samples, total_samples)
            pulse_len = end_idx - actual_start
            
            pulse_train[actual_start:end_idx] = pulse[:pulse_len]
        
        return pulse_train, time_vector
    
    def generate_matched_filter(self, 
                               waveform: np.ndarray,
                               filter_type: str = "matched",
                               window: Optional[str] = None,
                               sidelobe_control: bool = True) -> np.ndarray:
        """
        Generate matched filter reference for given waveform
        
        Args:
            waveform: Reference waveform
            filter_type: Type of filter ('matched', 'mismatched', 'adaptive')
            window: Optional window function for sidelobe control
            sidelobe_control: Enable sidelobe suppression
            
        Returns:
            Matched filter coefficients
        """
        # Create cache key
        cache_key = (id(waveform), filter_type, window, sidelobe_control)
        if cache_key in self._matched_filter_cache:
            return self._matched_filter_cache[cache_key]
        
        if filter_type == "matched":
            # Standard matched filter (time-reversed complex conjugate)
            matched_filter = np.conj(waveform[::-1])
            
        elif filter_type == "mismatched":
            # Mismatched filter for sidelobe suppression
            matched_filter = self._design_mismatched_filter(waveform)
            
        elif filter_type == "adaptive":
            # Adaptive filter design based on expected interference
            matched_filter = self._design_adaptive_filter(waveform)
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply window for sidelobe control
        if window and sidelobe_control:
            window_func = self._get_window_function(window, len(matched_filter))
            matched_filter *= window_func
        
        # Normalize filter
        matched_filter /= np.linalg.norm(matched_filter)
        
        # Cache result
        self._matched_filter_cache[cache_key] = matched_filter
        
        return matched_filter
    
    def apply_transmit_beamforming(self,
                                  waveform: np.ndarray,
                                  steering_angle: float,
                                  beamforming_type: BeamformingType = BeamformingType.CONVENTIONAL,
                                  sidelobe_level: float = -30.0) -> np.ndarray:
        """
        Apply transmit beamforming to waveform for phased arrays
        
        Args:
            waveform: Base waveform
            steering_angle: Beam steering angle in radians
            beamforming_type: Type of beamforming algorithm
            sidelobe_level: Desired sidelobe level in dB
            
        Returns:
            Array of beamformed waveforms (one per element)
        """
        num_elements = self.array_geometry.num_elements
        
        # Calculate element positions if not provided
        if self.array_geometry.element_positions is None:
            positions = self._calculate_element_positions()
        else:
            positions = self.array_geometry.element_positions
        
        # Calculate steering phases
        wavelength = 3e8 / self.params.center_frequency if self.params.center_frequency > 0 else 0.1
        k = 2 * np.pi / wavelength
        
        # For linear array, steering phases based on x-position and angle
        steering_phases = k * positions[:, 0] * np.sin(steering_angle)
        
        # Calculate element weights based on beamforming type
        weights = self._calculate_beamforming_weights(beamforming_type, sidelobe_level)
        
        # Apply weights and phases to create element waveforms
        element_waveforms = np.zeros((num_elements, len(waveform)), dtype=complex)
        
        for i in range(num_elements):
            # Apply amplitude weight and steering phase
            element_waveforms[i] = waveform * weights[i] * np.exp(1j * steering_phases[i])
            
            # Add element-specific impairments
            element_waveforms[i] = self._apply_element_impairments(element_waveforms[i], i)
        
        return element_waveforms
    
    def calculate_beam_pattern(self,
                              angles: np.ndarray,
                              frequency: float,
                              weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate antenna beam pattern
        
        Args:
            angles: Array of angles in radians
            frequency: Operating frequency in Hz
            weights: Element weights (if None, uses uniform weighting)
            
        Returns:
            Normalized beam pattern
        """
        num_elements = self.array_geometry.num_elements
        
        if weights is None:
            weights = np.ones(num_elements)
        
        # Element positions
        if self.array_geometry.element_positions is None:
            positions = self._calculate_element_positions()
        else:
            positions = self.array_geometry.element_positions
        
        wavelength = 3e8 / frequency
        k = 2 * np.pi / wavelength
        
        # Calculate array factor
        beam_pattern = np.zeros(len(angles), dtype=complex)
        
        for angle in angles:
            phase_factors = np.exp(1j * k * positions[:, 0] * np.sin(angle))
            beam_pattern[angles == angle] = np.sum(weights * phase_factors)
        
        # Normalize to peak
        beam_pattern = np.abs(beam_pattern)
        beam_pattern /= np.max(beam_pattern)
        
        return beam_pattern
    
    def _apply_pulse_variations(self, pulse: np.ndarray, pulse_index: int) -> np.ndarray:
        """Apply pulse-to-pulse variations"""
        # Amplitude variations (thermal drift and random fluctuations)
        thermal_factor = 1.0 + self._thermal_state * 0.01
        random_factor = 1.0 + np.random.normal(0, 0.005)
        amplitude_factor = thermal_factor * random_factor
        
        # Phase variations (oscillator drift)
        phase_drift = np.random.normal(0, 0.1) * pulse_index  # Cumulative drift
        random_phase = np.random.normal(0, 0.05)  # Random phase noise
        phase_factor = np.exp(1j * (phase_drift + random_phase))
        
        return pulse * amplitude_factor * phase_factor
    
    def _apply_hardware_impairments(self, waveform: np.ndarray) -> np.ndarray:
        """Apply realistic hardware impairments to waveform"""
        impaired_waveform = waveform.copy()
        
        # Phase noise
        phase_noise = self._generate_phase_noise(len(waveform))
        impaired_waveform *= np.exp(1j * phase_noise)
        
        # Amplitude ripple and drift
        amplitude_ripple = 1.0 + self.hardware_impairments.amplitude_ripple * \
                          np.sin(2 * np.pi * np.arange(len(waveform)) / len(waveform) * 3)
        amplitude_drift = 1.0 + self.hardware_impairments.amplitude_drift * \
                         np.random.normal(0, 0.1)
        impaired_waveform *= amplitude_ripple * amplitude_drift
        
        # I/Q imbalance
        i_component = np.real(impaired_waveform)
        q_component = np.imag(impaired_waveform)
        
        # Gain imbalance
        q_component *= (1.0 + self.hardware_impairments.iq_imbalance_gain)
        
        # Phase imbalance
        phase_imbalance_rad = np.deg2rad(self.hardware_impairments.iq_imbalance_phase)
        i_corrected = i_component
        q_corrected = q_component * np.cos(phase_imbalance_rad) + \
                     i_component * np.sin(phase_imbalance_rad)
        
        # DC offsets
        i_corrected += self.hardware_impairments.dc_offset_i
        q_corrected += self.hardware_impairments.dc_offset_q
        
        impaired_waveform = i_corrected + 1j * q_corrected
        
        return impaired_waveform
    
    def _apply_power_amplifier(self, waveform: np.ndarray) -> np.ndarray:
        """Apply power amplifier nonlinearities and thermal effects"""
        # Calculate instantaneous power
        instantaneous_power = np.abs(waveform)**2
        avg_power = np.mean(instantaneous_power)
        
        # Update thermal state (simplified thermal model)
        power_delta = avg_power - self._previous_power
        thermal_rate = 1.0 / self.pa_model.thermal_time_constant
        self._thermal_state += power_delta * thermal_rate * (1.0 / self.params.sample_rate)
        self._thermal_state *= np.exp(-thermal_rate * (1.0 / self.params.sample_rate))
        self._previous_power = avg_power
        
        # Apply AM-AM compression
        max_power_linear = self.pa_model.max_power
        p1db_linear = 10**(self.pa_model.p1db_compression / 10) * max_power_linear
        
        # Rapp model for AM-AM compression
        smoothness = 2.0  # Smoothness parameter
        gain = 10**(self.pa_model.gain / 10)
        
        compressed_amplitude = np.zeros_like(np.abs(waveform))
        for i, amp in enumerate(np.abs(waveform)):
            input_power = amp**2
            if input_power > 0:
                output_power = input_power * gain / \
                              (1 + (input_power * gain / p1db_linear)**smoothness)**(1/smoothness)
                compressed_amplitude[i] = np.sqrt(output_power)
            else:
                compressed_amplitude[i] = 0
        
        # Apply AM-PM conversion
        phase_shift = self.pa_model.phase_shift_coefficient * \
                     (20 * np.log10(compressed_amplitude / np.abs(waveform) + 1e-12))
        phase_shift_rad = np.deg2rad(phase_shift)
        
        # Combine amplitude compression and phase shift
        pa_output = compressed_amplitude * np.exp(1j * (np.angle(waveform) + phase_shift_rad))
        
        # Add harmonics (simplified - only fundamental frequency effects)
        # In a real implementation, this would require upsampling and filtering
        
        # Add thermal noise
        thermal_noise_power = 1e-15 * (1.0 + self._thermal_state)  # Simplified
        thermal_noise = np.random.normal(0, np.sqrt(thermal_noise_power/2), len(waveform)) + \
                       1j * np.random.normal(0, np.sqrt(thermal_noise_power/2), len(waveform))
        
        return pa_output + thermal_noise
    
    def _generate_phase_noise(self, num_samples: int) -> np.ndarray:
        """Generate realistic phase noise"""
        # Create frequency vector
        fs = self.params.sample_rate
        freqs = np.fft.fftfreq(num_samples, 1/fs)[1:num_samples//2]  # Exclude DC
        
        # Phase noise PSD model: L(f) = L0 / f^alpha
        L0 = 10**(self.hardware_impairments.phase_noise_level / 10)
        alpha = -self.hardware_impairments.phase_noise_slope / 10  # Convert to linear slope
        
        psd = L0 / (freqs**alpha)
        
        # Generate white noise and shape it
        white_noise = np.random.randn(len(freqs))
        shaped_noise_freq = white_noise * np.sqrt(psd * fs)
        
        # Convert to time domain (single-sided spectrum)
        noise_freq = np.zeros(num_samples, dtype=complex)
        noise_freq[1:num_samples//2] = shaped_noise_freq
        noise_freq[num_samples//2+1:] = np.conj(shaped_noise_freq[::-1])
        
        phase_noise = np.real(np.fft.ifft(noise_freq))
        
        return phase_noise
    
    def _design_mismatched_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Design mismatched filter for sidelobe suppression"""
        # Simplified mismatched filter design using least squares
        N = len(waveform)
        
        # Create toeplitz matrix for autocorrelation
        autocorr = np.correlate(waveform, waveform, mode='full')
        mid_idx = len(autocorr) // 2
        
        # Desired response (low sidelobes)
        desired = np.zeros(N)
        desired[N//2] = 1.0  # Peak at center
        
        # Design filter to minimize sidelobes while maintaining gain
        # This is a simplified approach - real implementation would use more sophisticated methods
        matched_filter = np.conj(waveform[::-1])
        
        # Apply slight tapering to reduce sidelobes
        taper = signal.windows.taylor(N, nbar=4, sll=-35)
        matched_filter *= taper
        
        return matched_filter
    
    def _design_adaptive_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Design adaptive matched filter"""
        # Simplified adaptive filter - in practice this would adapt to interference
        return np.conj(waveform[::-1])
    
    def _calculate_element_positions(self) -> np.ndarray:
        """Calculate element positions for array geometry"""
        num_elements = self.array_geometry.num_elements
        spacing = self.array_geometry.element_spacing
        
        if self.array_geometry.array_type == "linear":
            # Linear array along x-axis
            x_positions = np.arange(num_elements) * spacing
            x_positions -= np.mean(x_positions)  # Center array
            positions = np.zeros((num_elements, 3))
            positions[:, 0] = x_positions
            
        elif self.array_geometry.array_type == "planar":
            # Rectangular planar array
            n_rows = int(np.sqrt(num_elements))
            n_cols = num_elements // n_rows
            
            positions = np.zeros((num_elements, 3))
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    if idx < num_elements:
                        positions[idx, 0] = (j - n_cols/2) * spacing
                        positions[idx, 1] = (i - n_rows/2) * spacing
                        idx += 1
                        
        elif self.array_geometry.array_type == "circular":
            # Circular array
            angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False)
            radius = spacing * num_elements / (2 * np.pi)
            
            positions = np.zeros((num_elements, 3))
            positions[:, 0] = radius * np.cos(angles)
            positions[:, 1] = radius * np.sin(angles)
            
        else:
            # Default to linear array
            positions = np.zeros((num_elements, 3))
            positions[:, 0] = np.arange(num_elements) * spacing
        
        return positions
    
    def _calculate_beamforming_weights(self, 
                                     beamforming_type: BeamformingType,
                                     sidelobe_level: float) -> np.ndarray:
        """Calculate beamforming weights"""
        num_elements = self.array_geometry.num_elements
        
        if beamforming_type == BeamformingType.CONVENTIONAL:
            # Uniform weighting
            weights = np.ones(num_elements)
            
        elif beamforming_type == BeamformingType.TAYLOR:
            # Taylor weighting for sidelobe control
            weights = signal.windows.taylor(num_elements, nbar=4, sll=sidelobe_level)
            
        elif beamforming_type == BeamformingType.CHEBYSHEV:
            # Chebyshev weighting
            weights = signal.windows.chebwin(num_elements, at=-sidelobe_level)
            
        elif beamforming_type == BeamformingType.ADAPTIVE:
            # Simplified adaptive weights (would require interference covariance matrix)
            weights = np.ones(num_elements)
            # Add slight random perturbations
            weights += 0.1 * np.random.randn(num_elements)
            weights /= np.sum(weights)  # Normalize
            
        else:
            weights = np.ones(num_elements)
        
        return weights
    
    def _apply_element_impairments(self, waveform: np.ndarray, element_idx: int) -> np.ndarray:
        """Apply element-specific impairments"""
        # Element-specific amplitude and phase errors
        amp_error = 1.0 + np.random.normal(0, 0.02)  # 2% RMS amplitude error
        phase_error = np.random.normal(0, np.deg2rad(5))  # 5 degree RMS phase error
        
        # Element-specific gain and phase variations
        element_factor = amp_error * np.exp(1j * phase_error)
        
        return waveform * element_factor
    
    def _get_window_function(self, window_type: str, length: int) -> np.ndarray:
        """Get window function of specified type and length"""
        if window_type == "hamming":
            return np.hamming(length)
        elif window_type == "hann":
            return np.hanning(length)
        elif window_type == "blackman":
            return np.blackman(length)
        elif window_type == "taylor":
            return signal.windows.taylor(length, nbar=4, sll=-35)
        elif window_type == "chebyshev":
            return signal.windows.chebwin(length, at=60)
        else:
            return np.ones(length)
    
    def get_pulse_compression_gain(self, waveform: np.ndarray) -> float:
        """
        Calculate pulse compression gain
        
        Args:
            waveform: Input waveform
            
        Returns:
            Pulse compression gain in dB
        """
        # Time-bandwidth product
        time_bandwidth = self.params.pulse_width * self.params.bandwidth
        
        # Theoretical compression gain
        theoretical_gain = 10 * np.log10(time_bandwidth)
        
        # Account for losses due to weighting, mismatched filtering, etc.
        processing_loss = 1.0  # dB (simplified)
        
        actual_gain = theoretical_gain - processing_loss
        
        return actual_gain
    
    def analyze_waveform_properties(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Analyze key properties of generated waveform
        
        Args:
            waveform: Waveform to analyze
            
        Returns:
            Dictionary of waveform properties
        """
        properties = {}
        
        # Peak-to-average power ratio (PAPR)
        instantaneous_power = np.abs(waveform)**2
        peak_power = np.max(instantaneous_power)
        avg_power = np.mean(instantaneous_power)
        properties['papr_db'] = 10 * np.log10(peak_power / avg_power)
        
        # Bandwidth occupancy
        spectrum = np.fft.fft(waveform)
        power_spectrum = np.abs(spectrum)**2
        power_spectrum_db = 10 * np.log10(power_spectrum / np.max(power_spectrum))
        
        # 99% bandwidth
        total_power = np.sum(power_spectrum)
        cumulative_power = np.cumsum(np.fft.fftshift(power_spectrum))
        freq_indices = np.where((cumulative_power >= 0.005 * total_power) & 
                               (cumulative_power <= 0.995 * total_power))[0]
        if len(freq_indices) > 0:
            bw_99_bins = freq_indices[-1] - freq_indices[0]
            properties['bandwidth_99_hz'] = bw_99_bins * self.params.sample_rate / len(waveform)
        else:
            properties['bandwidth_99_hz'] = self.params.bandwidth
        
        # Autocorrelation properties
        autocorr = self.autocorrelation(waveform)
        properties['peak_sidelobe_ratio_db'] = self.peak_sidelobe_ratio(waveform)
        properties['integrated_sidelobe_ratio_db'] = self.integrated_sidelobe_ratio(waveform)
        
        # Range resolution
        properties['range_resolution_m'] = 3e8 / (2 * self.params.bandwidth)
        
        return properties