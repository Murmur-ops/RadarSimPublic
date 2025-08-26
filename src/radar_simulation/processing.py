"""
Range-Doppler processing for radar simulation

This module implements comprehensive range-Doppler processing algorithms that operate
exclusively on received signals without access to ground truth data. It includes
matched filtering, range/Doppler FFTs, coherent/non-coherent integration, and
MTI/MTD filtering for clutter suppression.

CRITICAL: All processing operates only on received signals - no ground truth access.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, fft2, fftshift, fftfreq
from scipy import interpolate
import warnings

# Import the existing SignalProcessor
from ..signal import SignalProcessor


class WindowType(Enum):
    """Window function types"""
    RECTANGULAR = "rectangular"
    HAMMING = "hamming"
    HANN = "hann"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    TAYLOR = "taylor"
    CHEBYSHEV = "chebyshev"


class IntegrationType(Enum):
    """Integration types"""
    COHERENT = "coherent"
    NON_COHERENT = "non_coherent"
    BINARY = "binary"


class MTIFilterType(Enum):
    """MTI filter types"""
    TWO_PULSE = "two_pulse"
    THREE_PULSE = "three_pulse"
    FOUR_PULSE = "four_pulse"
    STAGGERED = "staggered"


@dataclass
class ProcessingParameters:
    """Range-Doppler processing parameters"""
    # Basic parameters
    sample_rate: float  # Sample rate in Hz
    pulse_repetition_frequency: float  # PRF in Hz
    center_frequency: float  # Radar center frequency in Hz
    bandwidth: float  # Signal bandwidth in Hz
    
    # Range processing
    range_window: WindowType = WindowType.HAMMING
    range_fft_size: Optional[int] = None  # None = auto-size
    range_zero_padding_factor: int = 1
    
    # Doppler processing  
    doppler_window: WindowType = WindowType.HAMMING
    doppler_fft_size: Optional[int] = None  # None = auto-size
    coherent_processing_interval: float = 0.1  # CPI in seconds
    
    # Integration
    integration_type: IntegrationType = IntegrationType.COHERENT
    integration_pulses: int = 16
    
    # MTI/MTD
    mti_enabled: bool = True
    mti_filter_type: MTIFilterType = MTIFilterType.THREE_PULSE
    clutter_velocity: float = 0.0  # Clutter velocity in m/s
    
    # Detection
    cfar_enabled: bool = True
    cfar_guard_cells: int = 2
    cfar_training_cells: int = 16
    cfar_false_alarm_rate: float = 1e-6
    
    @property
    def speed_of_light(self) -> float:
        """Speed of light in m/s"""
        return 3e8
    
    @property
    def wavelength(self) -> float:
        """Radar wavelength in meters"""
        return self.speed_of_light / self.center_frequency
    
    @property
    def range_resolution(self) -> float:
        """Range resolution in meters"""
        return self.speed_of_light / (2 * self.bandwidth)
    
    @property
    def max_unambiguous_range(self) -> float:
        """Maximum unambiguous range in meters"""
        return self.speed_of_light / (2 * self.pulse_repetition_frequency)
    
    @property
    def max_unambiguous_velocity(self) -> float:
        """Maximum unambiguous velocity in m/s"""
        return self.wavelength * self.pulse_repetition_frequency / 4
    
    @property
    def velocity_resolution(self) -> float:
        """Velocity resolution in m/s"""
        cpi_seconds = self.coherent_processing_interval
        return self.wavelength / (4 * cpi_seconds)


class RangeDopplerProcessor:
    """
    Comprehensive range-Doppler processor for radar signals
    
    Implements the complete signal processing chain from raw I/Q samples to
    range-Doppler maps and detections. All processing operates exclusively on
    received signals without any access to ground truth target information.
    """
    
    def __init__(self, params: ProcessingParameters, reference_waveform: np.ndarray):
        """
        Initialize range-Doppler processor
        
        Args:
            params: Processing parameters
            reference_waveform: Reference waveform for matched filtering
        """
        self.params = params
        self.reference_waveform = reference_waveform
        
        # Initialize SignalProcessor for core algorithms
        self.signal_processor = SignalProcessor(
            sample_rate=params.sample_rate,
            bandwidth=params.bandwidth
        )
        
        # Pre-compute window functions
        self._range_window = None
        self._doppler_window = None
        self._mti_filter_coeffs = None
        
        # Processing state
        self._previous_pulses = []  # For MTI filtering
        self._noise_floor_estimate = None
        
        # Pre-compute derived parameters
        self._setup_processing_parameters()
    
    def process_pulse_train(self, 
                           pulse_data: np.ndarray,
                           timestamps: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Process a train of radar pulses to generate range-Doppler map
        
        Args:
            pulse_data: 2D array of pulse samples (num_range_samples, num_pulses)
            timestamps: Pulse timestamps in seconds (optional)
            
        Returns:
            Dictionary containing:
                - 'range_doppler_map': Range-Doppler map in dB
                - 'range_compressed': Range-compressed pulses
                - 'mti_filtered': MTI filtered data
                - 'detections': Detection results
                - 'range_bins': Range bin values in meters
                - 'velocity_bins': Velocity bin values in m/s
                - 'noise_floor': Estimated noise floor
        """
        if pulse_data.ndim != 2:
            raise ValueError("pulse_data must be 2D array (range_samples, num_pulses)")
        
        num_range_samples, num_pulses = pulse_data.shape
        
        # Step 1: Range compression (matched filtering)
        range_compressed = self._range_compression(pulse_data)
        
        # Step 2: MTI filtering for clutter suppression
        if self.params.mti_enabled and num_pulses > 1:
            mti_filtered = self._mti_filtering(range_compressed)
        else:
            mti_filtered = range_compressed
        
        # Step 3: Doppler processing
        doppler_processed = self._doppler_processing(mti_filtered)
        
        # Step 4: Integration
        integrated = self._signal_integration(doppler_processed)
        
        # Step 5: Generate range-Doppler map
        rd_map, range_bins, velocity_bins = self._generate_range_doppler_map(integrated)
        
        # Step 6: Noise floor estimation
        noise_floor = self._estimate_noise_floor(rd_map)
        
        # Step 7: Detection processing
        detections = {}
        if self.params.cfar_enabled:
            detections = self._cfar_detection(rd_map, noise_floor)
        
        return {
            'range_doppler_map': rd_map,
            'range_compressed': range_compressed,
            'mti_filtered': mti_filtered,
            'doppler_processed': doppler_processed,
            'integrated': integrated,
            'detections': detections,
            'range_bins': range_bins,
            'velocity_bins': velocity_bins,
            'noise_floor': noise_floor
        }
    
    def _range_compression(self, pulse_data: np.ndarray) -> np.ndarray:
        """
        Perform range compression using matched filtering
        
        Args:
            pulse_data: Raw pulse data
            
        Returns:
            Range-compressed data
        """
        num_range_samples, num_pulses = pulse_data.shape
        compressed_data = np.zeros_like(pulse_data, dtype=complex)
        
        # Apply matched filter to each pulse
        for pulse_idx in range(num_pulses):
            pulse = pulse_data[:, pulse_idx]
            
            # Matched filtering using SignalProcessor
            compressed_pulse = self.signal_processor.matched_filter(
                pulse, self.reference_waveform
            )
            
            # Handle size differences due to convolution
            if len(compressed_pulse) == len(pulse):
                compressed_data[:, pulse_idx] = compressed_pulse
            else:
                # Truncate or pad to match original size
                if len(compressed_pulse) > len(pulse):
                    start_idx = (len(compressed_pulse) - len(pulse)) // 2
                    compressed_data[:, pulse_idx] = compressed_pulse[start_idx:start_idx + len(pulse)]
                else:
                    pad_before = (len(pulse) - len(compressed_pulse)) // 2
                    pad_after = len(pulse) - len(compressed_pulse) - pad_before
                    compressed_data[:, pulse_idx] = np.pad(compressed_pulse, (pad_before, pad_after))
        
        # Apply range windowing
        if self._range_window is not None:
            compressed_data = compressed_data * self._range_window[:, np.newaxis]
        
        return compressed_data
    
    def _mti_filtering(self, pulse_data: np.ndarray) -> np.ndarray:
        """
        Apply Moving Target Indicator (MTI) filtering
        
        Args:
            pulse_data: Input pulse data
            
        Returns:
            MTI filtered data
        """
        if self.params.mti_filter_type == MTIFilterType.TWO_PULSE:
            return self.signal_processor.mti_filter(pulse_data, "two_pulse")
        
        elif self.params.mti_filter_type == MTIFilterType.THREE_PULSE:
            return self.signal_processor.mti_filter(pulse_data, "three_pulse")
        
        elif self.params.mti_filter_type == MTIFilterType.FOUR_PULSE:
            # Four-pulse canceller
            if pulse_data.shape[1] < 4:
                warnings.warn("Insufficient pulses for four-pulse MTI, using available pulses")
                return self.signal_processor.mti_filter(pulse_data, "three_pulse")
            
            # Four-pulse MTI: y[n] = x[n] - 3*x[n-1] + 3*x[n-2] - x[n-3]
            mti_output = np.zeros((pulse_data.shape[0], pulse_data.shape[1] - 3), dtype=complex)
            mti_output = (pulse_data[:, 3:] - 3 * pulse_data[:, 2:-1] + 
                         3 * pulse_data[:, 1:-2] - pulse_data[:, :-3])
            return mti_output
        
        elif self.params.mti_filter_type == MTIFilterType.STAGGERED:
            # Simplified staggered PRF MTI
            return self._staggered_mti(pulse_data)
        
        else:
            raise ValueError(f"Unknown MTI filter type: {self.params.mti_filter_type}")
    
    def _staggered_mti(self, pulse_data: np.ndarray) -> np.ndarray:
        """
        Staggered PRF MTI filter implementation
        
        Args:
            pulse_data: Input pulse data
            
        Returns:
            Staggered MTI filtered data
        """
        # Simplified implementation - alternating PRF
        num_pulses = pulse_data.shape[1]
        if num_pulses < 4:
            return self.signal_processor.mti_filter(pulse_data, "two_pulse")
        
        # Process even and odd pulses separately
        even_pulses = pulse_data[:, ::2]
        odd_pulses = pulse_data[:, 1::2]
        
        # Apply MTI to each stagger
        even_mti = self.signal_processor.mti_filter(even_pulses, "two_pulse")
        odd_mti = self.signal_processor.mti_filter(odd_pulses, "two_pulse")
        
        # Combine results (simplified)
        min_pulses = min(even_mti.shape[1], odd_mti.shape[1])
        combined = (even_mti[:, :min_pulses] + odd_mti[:, :min_pulses]) / 2
        
        return combined
    
    def _doppler_processing(self, pulse_data: np.ndarray) -> np.ndarray:
        """
        Perform Doppler processing using FFT
        
        Args:
            pulse_data: MTI filtered pulse data
            
        Returns:
            Doppler processed data
        """
        num_range_bins, num_pulses = pulse_data.shape
        
        # Determine FFT size
        if self.params.doppler_fft_size is None:
            fft_size = num_pulses
        else:
            fft_size = self.params.doppler_fft_size
        
        # Apply Doppler window
        if self._doppler_window is not None:
            windowed_data = pulse_data * self._doppler_window[np.newaxis, :num_pulses]
        else:
            windowed_data = pulse_data
        
        # Zero-pad if necessary
        if fft_size > num_pulses:
            pad_width = ((0, 0), (0, fft_size - num_pulses))
            windowed_data = np.pad(windowed_data, pad_width, mode='constant')
        
        # Perform Doppler FFT
        doppler_spectrum = fft(windowed_data, n=fft_size, axis=1)
        doppler_spectrum = fftshift(doppler_spectrum, axes=1)
        
        return doppler_spectrum
    
    def _signal_integration(self, doppler_data: np.ndarray) -> np.ndarray:
        """
        Perform signal integration
        
        Args:
            doppler_data: Doppler processed data
            
        Returns:
            Integrated data
        """
        if self.params.integration_type == IntegrationType.COHERENT:
            return self.signal_processor.coherent_integration(doppler_data)
        
        elif self.params.integration_type == IntegrationType.NON_COHERENT:
            return self.signal_processor.non_coherent_integration(doppler_data)
        
        elif self.params.integration_type == IntegrationType.BINARY:
            # Binary integration (detection counting)
            threshold = np.percentile(np.abs(doppler_data), 95)  # Adaptive threshold
            detections = np.abs(doppler_data) > threshold
            return np.sum(detections.astype(float), axis=1)
        
        else:
            raise ValueError(f"Unknown integration type: {self.params.integration_type}")
    
    def _generate_range_doppler_map(self, integrated_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate range-Doppler map from integrated data
        
        Args:
            integrated_data: Integrated signal data
            
        Returns:
            Tuple of (RD map in dB, range bins, velocity bins)
        """
        if integrated_data.ndim == 1:
            # Single range profile - expand for consistency
            rd_map = integrated_data[:, np.newaxis]
        else:
            rd_map = integrated_data
        
        # Convert to magnitude in dB
        rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-10)
        
        # Generate range bins
        num_range_bins = rd_map.shape[0]
        range_bins = np.arange(num_range_bins) * self.params.range_resolution
        
        # Generate velocity bins
        num_doppler_bins = rd_map.shape[1]
        doppler_freqs = fftshift(fftfreq(num_doppler_bins, 1/self.params.pulse_repetition_frequency))
        velocity_bins = doppler_freqs * self.params.wavelength / 2
        
        return rd_map_db, range_bins, velocity_bins
    
    def _estimate_noise_floor(self, rd_map: np.ndarray) -> float:
        """
        Estimate noise floor from range-Doppler map
        
        Args:
            rd_map: Range-Doppler map in dB
            
        Returns:
            Estimated noise floor in dB
        """
        # Use lower percentile to estimate noise floor
        noise_floor = np.percentile(rd_map, 10)
        self._noise_floor_estimate = noise_floor
        return noise_floor
    
    def _cfar_detection(self, rd_map: np.ndarray, noise_floor: float) -> Dict[str, np.ndarray]:
        """
        Perform CFAR detection on range-Doppler map
        
        Args:
            rd_map: Range-Doppler map in dB
            noise_floor: Estimated noise floor
            
        Returns:
            Dictionary containing detection results
        """
        # Convert back to linear scale for CFAR
        rd_map_linear = 10**(rd_map / 20)
        
        detections = {}
        
        if rd_map.ndim == 1:
            # 1D CFAR for range profile
            det_mask, threshold = self.signal_processor.cfar_detector(
                rd_map_linear,
                num_guard=self.params.cfar_guard_cells,
                num_train=self.params.cfar_training_cells,
                pfa=self.params.cfar_false_alarm_rate
            )
            
            detections['range_detections'] = det_mask
            detections['range_threshold'] = 20 * np.log10(threshold + 1e-10)
            
        else:
            # 2D CFAR for range-Doppler map
            det_mask = self._cfar_2d(rd_map_linear)
            detections['rd_detections'] = det_mask
            
            # Extract detection coordinates
            det_indices = np.where(det_mask)
            if len(det_indices[0]) > 0:
                detections['detection_ranges'] = det_indices[0] * self.params.range_resolution
                range_bins = np.arange(rd_map.shape[0]) * self.params.range_resolution
                velocity_bins = self._get_velocity_bins(rd_map.shape[1])
                detections['detection_velocities'] = velocity_bins[det_indices[1]]
                detections['detection_amplitudes'] = rd_map[det_indices]
            else:
                detections['detection_ranges'] = np.array([])
                detections['detection_velocities'] = np.array([])
                detections['detection_amplitudes'] = np.array([])
        
        return detections
    
    def _cfar_2d(self, rd_map: np.ndarray) -> np.ndarray:
        """
        2D CFAR detector implementation
        
        Args:
            rd_map: 2D range-Doppler map (linear scale)
            
        Returns:
            Boolean detection mask
        """
        num_range, num_doppler = rd_map.shape
        detections = np.zeros_like(rd_map, dtype=bool)
        
        guard = self.params.cfar_guard_cells
        train = self.params.cfar_training_cells
        pfa = self.params.cfar_false_alarm_rate
        
        # CFAR scaling factor
        num_training_cells = 8 * train * (train + guard)  # Approximate for 2D
        alpha = num_training_cells * (pfa**(-1/num_training_cells) - 1)
        
        # Process each cell
        for r in range(guard + train, num_range - guard - train):
            for d in range(guard + train, num_doppler - guard - train):
                
                # Define training region (excluding guard cells and CUT)
                training_region = []
                
                # Add training cells around the cell under test
                for dr in range(-train - guard, train + guard + 1):
                    for dd in range(-train - guard, train + guard + 1):
                        rr, dd_idx = r + dr, d + dd
                        
                        # Skip guard cells and cell under test
                        if abs(dr) <= guard and abs(dd) <= guard:
                            continue
                        
                        if 0 <= rr < num_range and 0 <= dd_idx < num_doppler:
                            training_region.append(rd_map[rr, dd_idx])
                
                if len(training_region) > 0:
                    # Estimate noise power
                    noise_power = np.mean(training_region)
                    
                    # Adaptive threshold
                    threshold = alpha * noise_power
                    
                    # Detection test
                    if rd_map[r, d] > threshold:
                        detections[r, d] = True
        
        return detections
    
    def _get_velocity_bins(self, num_doppler_bins: int) -> np.ndarray:
        """
        Get velocity bins for given number of Doppler bins
        
        Args:
            num_doppler_bins: Number of Doppler bins
            
        Returns:
            Velocity bins in m/s
        """
        doppler_freqs = fftshift(fftfreq(num_doppler_bins, 1/self.params.pulse_repetition_frequency))
        return doppler_freqs * self.params.wavelength / 2
    
    def _setup_processing_parameters(self) -> None:
        """Setup processing parameters and pre-compute windows"""
        
        # Setup range window
        if self.params.range_fft_size is None:
            range_size = len(self.reference_waveform)
        else:
            range_size = self.params.range_fft_size
            
        self._range_window = self._generate_window(
            self.params.range_window, range_size
        )
        
        # Setup Doppler window
        cpi_pulses = int(self.params.coherent_processing_interval * 
                        self.params.pulse_repetition_frequency)
        
        if self.params.doppler_fft_size is None:
            doppler_size = cpi_pulses
        else:
            doppler_size = self.params.doppler_fft_size
            
        self._doppler_window = self._generate_window(
            self.params.doppler_window, min(doppler_size, cpi_pulses)
        )
    
    def _generate_window(self, window_type: WindowType, size: int) -> np.ndarray:
        """
        Generate window function
        
        Args:
            window_type: Type of window
            size: Window size
            
        Returns:
            Window coefficients
        """
        if window_type == WindowType.RECTANGULAR:
            return np.ones(size)
        elif window_type == WindowType.HAMMING:
            return np.hamming(size)
        elif window_type == WindowType.HANN:
            return np.hann(size)
        elif window_type == WindowType.BLACKMAN:
            return np.blackman(size)
        elif window_type == WindowType.KAISER:
            return np.kaiser(size, beta=5)
        elif window_type == WindowType.TAYLOR:
            return signal.windows.taylor(size, nbar=4, sll=-35)
        elif window_type == WindowType.CHEBYSHEV:
            return signal.windows.chebwin(size, at=60)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    def get_processing_specs(self) -> Dict[str, Union[float, int, str]]:
        """
        Get processing specifications
        
        Returns:
            Dictionary of processing specifications
        """
        return {
            'range_resolution_m': self.params.range_resolution,
            'velocity_resolution_ms': self.params.velocity_resolution,
            'max_unambiguous_range_m': self.params.max_unambiguous_range,
            'max_unambiguous_velocity_ms': self.params.max_unambiguous_velocity,
            'coherent_processing_interval_s': self.params.coherent_processing_interval,
            'pulse_repetition_frequency_hz': self.params.pulse_repetition_frequency,
            'center_frequency_hz': self.params.center_frequency,
            'bandwidth_hz': self.params.bandwidth,
            'wavelength_m': self.params.wavelength,
            'mti_filter_type': self.params.mti_filter_type.value,
            'integration_type': self.params.integration_type.value,
            'range_window': self.params.range_window.value,
            'doppler_window': self.params.doppler_window.value
        }
    
    def estimate_snr(self, rd_map: np.ndarray, detection_coords: Tuple[int, int]) -> float:
        """
        Estimate SNR for a detection
        
        Args:
            rd_map: Range-Doppler map in dB
            detection_coords: (range_bin, doppler_bin) coordinates
            
        Returns:
            Estimated SNR in dB
        """
        r_idx, d_idx = detection_coords
        
        # Signal power
        signal_power_db = rd_map[r_idx, d_idx]
        
        # Noise power estimate
        if self._noise_floor_estimate is not None:
            noise_power_db = self._noise_floor_estimate
        else:
            noise_power_db = self._estimate_noise_floor(rd_map)
        
        return signal_power_db - noise_power_db
    
    def calculate_processing_gain(self) -> float:
        """
        Calculate total processing gain
        
        Returns:
            Processing gain in dB
        """
        # Matched filter gain
        mf_gain = 10 * np.log10(len(self.reference_waveform))
        
        # Coherent integration gain
        cpi_pulses = int(self.params.coherent_processing_interval * 
                        self.params.pulse_repetition_frequency)
        
        if self.params.integration_type == IntegrationType.COHERENT:
            integration_gain = 10 * np.log10(cpi_pulses)
        else:
            integration_gain = 10 * np.log10(np.sqrt(cpi_pulses))
        
        return mf_gain + integration_gain
    
    def get_clutter_spectrum(self, rd_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract clutter spectrum from range-Doppler map
        
        Args:
            rd_map: Range-Doppler map
            
        Returns:
            Tuple of (velocity bins, clutter spectrum)
        """
        # Average over range to get Doppler spectrum
        clutter_spectrum = np.mean(rd_map, axis=0)
        velocity_bins = self._get_velocity_bins(rd_map.shape[1])
        
        return velocity_bins, clutter_spectrum