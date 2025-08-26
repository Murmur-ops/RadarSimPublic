"""
Receiver chain modeling for radar simulation

This module implements a comprehensive receiver chain that models all the physical effects
and impairments found in real radar receivers, including thermal noise, antenna patterns,
RF front-end effects, ADC quantization, I/Q imbalance, and automatic gain control.

CRITICAL: All noise sources are truly random and independent of target truth data.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy import interpolate


class AntennaType(Enum):
    """Antenna pattern types"""
    ISOTROPIC = "isotropic"
    PARABOLIC = "parabolic"
    PHASED_ARRAY = "phased_array"
    HORN = "horn"
    CUSTOM = "custom"


@dataclass
class ReceiverParameters:
    """Receiver chain parameters"""
    # RF parameters
    noise_figure: float  # Noise figure in dB
    gain: float  # Receiver gain in dB
    bandwidth: float  # Receiver bandwidth in Hz
    center_frequency: float  # Center frequency in Hz
    
    # ADC parameters
    adc_bits: int  # ADC resolution in bits
    sample_rate: float  # ADC sample rate in Hz
    full_scale_voltage: float  # ADC full-scale input voltage
    
    # Antenna parameters
    antenna_gain: float  # Maximum antenna gain in dBi
    beamwidth_3db: float  # 3dB beamwidth in degrees
    sidelobe_level: float  # Peak sidelobe level in dB
    
    # RF front-end parameters
    compression_point: float  # 1dB compression point in dBm
    third_order_intercept: float  # Third-order intercept point in dBm
    
    # I/Q imbalance parameters
    iq_amplitude_imbalance: float  # I/Q amplitude imbalance in dB
    iq_phase_imbalance: float  # I/Q phase imbalance in degrees
    dc_offset_i: float  # DC offset on I channel in volts
    dc_offset_q: float  # DC offset on Q channel in volts
    
    # AGC parameters
    agc_enabled: bool = True
    agc_target_power: float = -20  # Target power level in dBm
    agc_time_constant: float = 1e-6  # AGC time constant in seconds
    agc_max_gain: float = 60  # Maximum AGC gain in dB
    agc_min_gain: float = 0  # Minimum AGC gain in dB


@dataclass
class NoiseParameters:
    """Thermal noise parameters"""
    temperature: float = 290  # System temperature in Kelvin
    boltzmann_constant: float = 1.38064852e-23  # Boltzmann constant
    
    @property
    def thermal_noise_power_density(self) -> float:
        """Thermal noise power density in W/Hz"""
        return self.boltzmann_constant * self.temperature


class ReceiverChain:
    """
    Comprehensive radar receiver chain simulation
    
    Models the complete signal path from antenna to digital output, including:
    - Antenna gain patterns with realistic sidelobes
    - Thermal noise generation (independent of target truth)
    - RF front-end effects (gain, compression, intermodulation)
    - I/Q downconversion with imbalance and DC offset
    - ADC quantization and sampling
    - Automatic gain control (AGC)
    """
    
    def __init__(self, params: ReceiverParameters):
        """
        Initialize receiver chain
        
        Args:
            params: Receiver parameters
        """
        self.params = params
        self.noise_params = NoiseParameters()
        
        # Initialize random number generator for repeatable but independent noise
        self._noise_rng = np.random.RandomState()
        
        # AGC state
        self._agc_gain = params.gain  # Current AGC gain
        self._agc_power_estimate = 0.0  # Current power estimate
        
        # Calculate derived parameters
        self._noise_power = self._calculate_noise_power()
        self._antenna_pattern = self._generate_antenna_pattern()
        
    def process_signal(self, 
                      signal_samples: np.ndarray,
                      signal_angles: Optional[np.ndarray] = None,
                      signal_powers: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Process signal through complete receiver chain
        
        Args:
            signal_samples: Complex baseband signal samples
            signal_angles: Arrival angles for each sample (degrees, optional)
            signal_powers: Signal powers for each sample (dBm, optional)
            
        Returns:
            Dictionary containing:
                - 'output': Final digital output samples
                - 'noise': Added noise samples
                - 'antenna_response': Antenna gain applied
                - 'agc_gain': AGC gain applied
                - 'quantized': Pre-quantization samples
        """
        # Initialize output dictionary
        output = {}
        
        # Step 1: Apply antenna gain pattern
        antenna_response = self._apply_antenna_pattern(signal_samples, signal_angles)
        output['antenna_response'] = antenna_response
        
        # Step 2: Add thermal noise (truly random, independent of signal)
        noise_samples = self._generate_thermal_noise(len(signal_samples))
        noisy_signal = antenna_response + noise_samples
        output['noise'] = noise_samples
        
        # Step 3: Apply RF front-end effects
        rf_output = self._apply_rf_frontend(noisy_signal, signal_powers)
        
        # Step 4: Apply AGC if enabled
        if self.params.agc_enabled:
            agc_output, agc_gain = self._apply_agc(rf_output)
            output['agc_gain'] = agc_gain
        else:
            agc_output = rf_output * (10**(self.params.gain/20))
            output['agc_gain'] = np.full(len(signal_samples), self.params.gain)
        
        # Step 5: Apply I/Q imbalance and DC offset
        iq_output = self._apply_iq_imbalance(agc_output)
        
        # Step 6: ADC quantization and sampling
        output['quantized'] = iq_output
        quantized_output = self._apply_adc_quantization(iq_output)
        
        output['output'] = quantized_output
        
        return output
    
    def _apply_antenna_pattern(self, 
                              signal: np.ndarray,
                              angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply antenna gain pattern to signal
        
        Args:
            signal: Input signal samples
            angles: Arrival angles in degrees (None = boresight)
            
        Returns:
            Signal with antenna pattern applied
        """
        if angles is None:
            # Assume boresight (maximum gain)
            gain_linear = 10**(self.params.antenna_gain / 20)
            return signal * gain_linear
        
        # Apply angle-dependent gain
        gain_pattern = np.zeros_like(angles)
        for i, angle in enumerate(angles):
            gain_pattern[i] = self._get_antenna_gain(angle)
        
        gain_linear = 10**(gain_pattern / 20)
        return signal * gain_linear
    
    def _get_antenna_gain(self, angle: float) -> float:
        """
        Get antenna gain at specific angle
        
        Args:
            angle: Angle from boresight in degrees
            
        Returns:
            Antenna gain in dB
        """
        angle = abs(angle)  # Assume symmetric pattern
        
        if angle <= self.params.beamwidth_3db / 2:
            # Main beam - use cosine pattern approximation
            normalized_angle = angle / (self.params.beamwidth_3db / 2)
            gain = self.params.antenna_gain - 3 * normalized_angle**2
        else:
            # Sidelobes - realistic sidelobe structure
            # Use interpolation of generated pattern
            angle_rad = np.deg2rad(angle)
            pattern_angles = np.linspace(0, np.pi, len(self._antenna_pattern))
            gain = np.interp(angle_rad, pattern_angles, self._antenna_pattern)
        
        return max(gain, self.params.antenna_gain + self.params.sidelobe_level - 20)
    
    def _generate_antenna_pattern(self) -> np.ndarray:
        """
        Generate realistic antenna pattern with sidelobes
        
        Returns:
            Antenna pattern in dB vs angle
        """
        angles = np.linspace(0, np.pi, 1800)  # 0.1 degree resolution
        pattern = np.zeros_like(angles)
        
        # Main beam (Gaussian approximation)
        beamwidth_rad = np.deg2rad(self.params.beamwidth_3db)
        sigma = beamwidth_rad / (2 * np.sqrt(2 * np.log(2)))
        
        for i, angle in enumerate(angles):
            if angle <= beamwidth_rad:
                # Main beam
                pattern[i] = self.params.antenna_gain * np.exp(-(angle**2) / (2 * sigma**2))
            else:
                # Sidelobes - realistic envelope with nulls
                # First null approximation
                first_null = 1.2 * beamwidth_rad
                
                if angle < first_null:
                    # Transition region
                    pattern[i] = self.params.antenna_gain + self.params.sidelobe_level * 0.5
                else:
                    # Sidelobe region with 1/theta decay
                    sidelobe_envelope = self.params.sidelobe_level - 20 * np.log10(angle / first_null)
                    
                    # Add realistic sidelobe ripple
                    ripple = 5 * np.sin(10 * angle) * np.exp(-angle / (2 * beamwidth_rad))
                    pattern[i] = sidelobe_envelope + ripple
        
        return pattern
    
    def _generate_thermal_noise(self, num_samples: int) -> np.ndarray:
        """
        Generate thermal noise samples (truly random)
        
        Args:
            num_samples: Number of noise samples to generate
            
        Returns:
            Complex thermal noise samples
        """
        # Thermal noise power
        noise_power_watts = self._noise_power
        noise_std = np.sqrt(noise_power_watts / 2)  # Complex noise (I and Q)
        
        # Generate independent Gaussian noise for I and Q channels
        noise_i = self._noise_rng.normal(0, noise_std, num_samples)
        noise_q = self._noise_rng.normal(0, noise_std, num_samples)
        
        return noise_i + 1j * noise_q
    
    def _calculate_noise_power(self) -> float:
        """
        Calculate thermal noise power
        
        Returns:
            Noise power in watts
        """
        # Thermal noise power: kTB * NF
        thermal_power = (self.noise_params.thermal_noise_power_density * 
                        self.params.bandwidth)
        
        # Apply noise figure
        noise_figure_linear = 10**(self.params.noise_figure / 10)
        total_noise_power = thermal_power * noise_figure_linear
        
        return total_noise_power
    
    def _apply_rf_frontend(self, 
                          signal: np.ndarray,
                          signal_powers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply RF front-end effects (compression, intermodulation)
        
        Args:
            signal: Input signal
            signal_powers: Signal power levels in dBm
            
        Returns:
            Signal with RF effects applied
        """
        output = signal.copy()
        
        # Apply gain compression if signal power is high
        if signal_powers is not None:
            for i, power_dbm in enumerate(signal_powers):
                if power_dbm > self.params.compression_point:
                    # Simple compression model
                    excess_power = power_dbm - self.params.compression_point
                    compression_factor = 1 / (1 + 0.1 * excess_power)
                    output[i] *= compression_factor
        
        # Add third-order intermodulation distortion for strong signals
        signal_power = np.mean(np.abs(output)**2)
        signal_power_dbm = 10 * np.log10(signal_power * 1000)  # Convert to dBm
        
        if signal_power_dbm > self.params.third_order_intercept - 20:
            # Generate third-order distortion
            distortion_power = (signal_power_dbm - self.params.third_order_intercept) / 2
            distortion_amplitude = np.sqrt(10**(distortion_power / 10) / 1000)
            
            # Third-order terms (simplified)
            distortion = distortion_amplitude * (output * np.abs(output)**2) / (signal_power**1.5)
            output += distortion
        
        return output
    
    def _apply_agc(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply automatic gain control
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (AGC output signal, AGC gain array)
        """
        output = np.zeros_like(signal)
        agc_gains = np.zeros(len(signal))
        
        # AGC time constant in samples
        alpha = np.exp(-1 / (self.params.agc_time_constant * self.params.sample_rate))
        
        for i, sample in enumerate(signal):
            # Update power estimate with exponential smoothing
            instantaneous_power = np.abs(sample)**2
            instantaneous_power_dbm = 10 * np.log10(instantaneous_power * 1000 + 1e-15)
            
            if i == 0:
                self._agc_power_estimate = instantaneous_power_dbm
            else:
                self._agc_power_estimate = (alpha * self._agc_power_estimate + 
                                          (1 - alpha) * instantaneous_power_dbm)
            
            # Calculate required gain adjustment
            power_error = self.params.agc_target_power - self._agc_power_estimate
            
            # Update AGC gain with limits
            self._agc_gain = np.clip(self._agc_gain + 0.1 * power_error,
                                   self.params.agc_min_gain,
                                   self.params.agc_max_gain)
            
            # Apply gain
            gain_linear = 10**(self._agc_gain / 20)
            output[i] = sample * gain_linear
            agc_gains[i] = self._agc_gain
        
        return output, agc_gains
    
    def _apply_iq_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply I/Q imbalance and DC offset
        
        Args:
            signal: Complex input signal
            
        Returns:
            Signal with I/Q imbalance applied
        """
        # Extract I and Q components
        i_channel = np.real(signal)
        q_channel = np.imag(signal)
        
        # Apply amplitude imbalance
        amplitude_imbalance_linear = 10**(self.params.iq_amplitude_imbalance / 20)
        q_channel *= amplitude_imbalance_linear
        
        # Apply phase imbalance
        phase_imbalance_rad = np.deg2rad(self.params.iq_phase_imbalance)
        
        # Rotation matrix for phase imbalance
        i_corrected = i_channel
        q_corrected = (i_channel * np.sin(phase_imbalance_rad) + 
                      q_channel * np.cos(phase_imbalance_rad))
        
        # Add DC offsets
        i_corrected += self.params.dc_offset_i
        q_corrected += self.params.dc_offset_q
        
        return i_corrected + 1j * q_corrected
    
    def _apply_adc_quantization(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply ADC quantization effects
        
        Args:
            signal: Analog input signal
            
        Returns:
            Quantized digital signal
        """
        # Calculate quantization levels
        num_levels = 2**self.params.adc_bits
        lsb = 2 * self.params.full_scale_voltage / num_levels
        
        # Separate I and Q channels
        i_channel = np.real(signal)
        q_channel = np.imag(signal)
        
        # Clip to full-scale range
        i_clipped = np.clip(i_channel, 
                           -self.params.full_scale_voltage,
                           self.params.full_scale_voltage)
        q_clipped = np.clip(q_channel,
                           -self.params.full_scale_voltage, 
                           self.params.full_scale_voltage)
        
        # Quantize
        i_quantized = np.round(i_clipped / lsb) * lsb
        q_quantized = np.round(q_clipped / lsb) * lsb
        
        # Add quantization noise (uniform distribution)
        quantization_noise_std = lsb / np.sqrt(12)
        i_quantized += self._noise_rng.uniform(-lsb/2, lsb/2, len(signal))
        q_quantized += self._noise_rng.uniform(-lsb/2, lsb/2, len(signal))
        
        return i_quantized + 1j * q_quantized
    
    def set_noise_seed(self, seed: int) -> None:
        """
        Set random seed for repeatable noise generation
        
        Args:
            seed: Random seed
        """
        self._noise_rng.seed(seed)
    
    def get_noise_power_dbm(self) -> float:
        """
        Get thermal noise power in dBm
        
        Returns:
            Noise power in dBm
        """
        return 10 * np.log10(self._noise_power * 1000)
    
    def get_dynamic_range(self) -> float:
        """
        Calculate receiver dynamic range
        
        Returns:
            Dynamic range in dB
        """
        # Minimum detectable signal (noise floor + SNR margin)
        noise_floor_dbm = self.get_noise_power_dbm()
        min_signal_dbm = noise_floor_dbm + 10  # 10 dB SNR margin
        
        # Maximum signal (compression point)
        max_signal_dbm = self.params.compression_point
        
        return max_signal_dbm - min_signal_dbm
    
    def calculate_noise_figure(self) -> float:
        """
        Calculate effective noise figure including all receiver stages
        
        Returns:
            Effective noise figure in dB
        """
        # This is simplified - in practice would include cascade calculations
        return self.params.noise_figure
    
    def get_sensitivity(self, snr_required: float = 10) -> float:
        """
        Calculate receiver sensitivity
        
        Args:
            snr_required: Required SNR in dB
            
        Returns:
            Sensitivity in dBm
        """
        noise_floor_dbm = self.get_noise_power_dbm()
        return noise_floor_dbm + snr_required
    
    def plot_antenna_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get antenna pattern for plotting
        
        Returns:
            Tuple of (angles in degrees, pattern in dB)
        """
        angles_rad = np.linspace(0, np.pi, len(self._antenna_pattern))
        angles_deg = np.rad2deg(angles_rad)
        return angles_deg, self._antenna_pattern
    
    def get_receiver_specs(self) -> Dict[str, float]:
        """
        Get key receiver specifications
        
        Returns:
            Dictionary of receiver specifications
        """
        return {
            'noise_figure_db': self.params.noise_figure,
            'gain_db': self.params.gain,
            'bandwidth_hz': self.params.bandwidth,
            'sensitivity_dbm': self.get_sensitivity(),
            'dynamic_range_db': self.get_dynamic_range(),
            'noise_floor_dbm': self.get_noise_power_dbm(),
            'compression_point_dbm': self.params.compression_point,
            'antenna_gain_dbi': self.params.antenna_gain,
            'beamwidth_deg': self.params.beamwidth_3db,
            'adc_bits': self.params.adc_bits,
            'sample_rate_hz': self.params.sample_rate
        }