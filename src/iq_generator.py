"""
IQ data generator for radar simulation
Generates realistic I/Q data including targets, noise, clutter, and receiver effects
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
import h5py
import struct
import os
from scipy import signal as scipy_signal
import json

from .radar import Radar, RadarParameters
from .target import Target
from .environment import Environment
from .waveforms import WaveformGenerator, WaveformParameters, WaveformType


@dataclass
class IQParameters:
    """Parameters for IQ data generation"""
    adc_bits: int = 14  # ADC resolution
    adc_max_voltage: float = 1.0  # ADC maximum input voltage
    if_frequency: float = 0  # Intermediate frequency (0 for baseband)
    iq_imbalance_amplitude: float = 0.01  # I/Q amplitude imbalance (ratio)
    iq_imbalance_phase: float = 0.01  # I/Q phase imbalance (radians)
    phase_noise_level: float = -80  # Phase noise level in dBc/Hz
    dc_offset_i: float = 0.0  # DC offset in I channel
    dc_offset_q: float = 0.0  # DC offset in Q channel
    enable_quantization: bool = True  # Enable ADC quantization
    enable_phase_noise: bool = True  # Enable phase noise
    enable_iq_imbalance: bool = True  # Enable I/Q imbalance


class IQDataGenerator:
    """Generate realistic IQ data for radar simulation"""
    
    def __init__(self, 
                 radar: Radar,
                 waveform_generator: WaveformGenerator,
                 iq_params: Optional[IQParameters] = None):
        """
        Initialize IQ data generator
        
        Args:
            radar: Radar system object
            waveform_generator: Waveform generator
            iq_params: IQ generation parameters
        """
        self.radar = radar
        self.waveform_gen = waveform_generator
        self.iq_params = iq_params or IQParameters()
        
        # Calculate derived parameters
        self.adc_levels = 2**self.iq_params.adc_bits
        self.adc_scale = self.iq_params.adc_max_voltage
        
    def generate_cpi(self,
                    targets: List[Target],
                    num_pulses: int,
                    pri: Optional[float] = None,
                    waveform_type: WaveformType = WaveformType.LFM,
                    environment: Optional[Environment] = None,
                    add_clutter: bool = True,
                    **waveform_kwargs) -> np.ndarray:
        """
        Generate Coherent Processing Interval (CPI) of IQ data
        
        Args:
            targets: List of targets
            num_pulses: Number of pulses in CPI
            pri: Pulse Repetition Interval (if None, uses radar PRF)
            waveform_type: Type of waveform to use
            environment: Environmental conditions
            add_clutter: Whether to add clutter
            **waveform_kwargs: Additional waveform parameters
            
        Returns:
            Complex IQ data array (fast_time_samples x num_pulses)
        """
        if pri is None:
            pri = 1.0 / self.radar.params.prf
        
        # Generate transmit waveform
        tx_waveform = self.waveform_gen.generate(waveform_type, **waveform_kwargs)
        num_samples = len(tx_waveform)
        
        # Initialize CPI data array
        cpi_data = np.zeros((num_samples, num_pulses), dtype=complex)
        
        # Generate each pulse
        for pulse_idx in range(num_pulses):
            # Update target positions (for moving targets)
            for target in targets:
                if pulse_idx > 0:
                    target.motion.update(pri)
            
            # Generate single pulse IQ data
            pulse_iq = self.generate_single_pulse(
                tx_waveform,
                targets,
                environment,
                add_clutter
            )
            
            cpi_data[:, pulse_idx] = pulse_iq
        
        return cpi_data
    
    def generate_single_pulse(self,
                            tx_waveform: np.ndarray,
                            targets: List[Target],
                            environment: Optional[Environment] = None,
                            add_clutter: bool = True) -> np.ndarray:
        """
        Generate IQ data for a single pulse
        
        Args:
            tx_waveform: Transmit waveform
            targets: List of targets
            environment: Environmental conditions
            add_clutter: Whether to add clutter
            
        Returns:
            Complex IQ data for single pulse
        """
        num_samples = len(tx_waveform)
        sample_rate = self.waveform_gen.params.sample_rate
        
        # Initialize received signal
        rx_signal = np.zeros(num_samples, dtype=complex)
        
        # Add target returns
        for target in targets:
            # Calculate delay and Doppler
            target_range = target.motion.range
            delay_samples = int(2 * target_range / 3e8 * sample_rate)
            
            if delay_samples < num_samples:
                # Calculate received power
                rcs = target.get_rcs()
                rx_power = self.radar.radar_equation(target_range, rcs)
                amplitude = np.sqrt(rx_power)
                
                # Apply Doppler shift
                doppler = self.radar.doppler_shift(target.get_radial_velocity())
                doppler_phase = 2 * np.pi * doppler * np.arange(num_samples) / sample_rate
                
                # Apply propagation effects
                if environment:
                    prop_factor = environment.propagation_factor(
                        target_range,
                        self.radar.params.frequency,
                        target.motion.position[2]
                    )
                    amplitude *= prop_factor
                
                # Create delayed and Doppler-shifted return
                target_return = np.zeros(num_samples, dtype=complex)
                if delay_samples > 0:
                    target_return[delay_samples:] = (
                        amplitude * tx_waveform[:-delay_samples] * 
                        np.exp(1j * doppler_phase[delay_samples:])
                    )
                else:
                    target_return = amplitude * tx_waveform * np.exp(1j * doppler_phase)
                
                rx_signal += target_return
        
        # Add clutter
        if add_clutter and environment:
            clutter = self._generate_clutter(num_samples, sample_rate, environment)
            rx_signal += clutter
        
        # Add thermal noise
        noise = self._generate_noise(num_samples)
        rx_signal += noise
        
        # Apply receiver effects
        if self.iq_params.enable_iq_imbalance:
            rx_signal = self._apply_iq_imbalance(rx_signal)
        
        if self.iq_params.enable_phase_noise:
            rx_signal = self._apply_phase_noise(rx_signal)
        
        # Add DC offset
        rx_signal += complex(self.iq_params.dc_offset_i, self.iq_params.dc_offset_q)
        
        # Quantization (ADC)
        if self.iq_params.enable_quantization:
            rx_signal = self._quantize_signal(rx_signal)
        
        return rx_signal
    
    def _generate_noise(self, num_samples: int) -> np.ndarray:
        """
        Generate thermal noise
        
        Args:
            num_samples: Number of samples
            
        Returns:
            Complex noise samples
        """
        # Calculate noise power
        k_boltzmann = 1.38e-23
        temperature = 290
        noise_figure = 10**(self.radar.params.noise_figure / 10)
        bandwidth = self.waveform_gen.params.bandwidth
        
        noise_power = k_boltzmann * temperature * bandwidth * noise_figure
        noise_amplitude = np.sqrt(noise_power / 2)  # Split between I and Q
        
        # Generate complex Gaussian noise
        noise = noise_amplitude * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )
        
        return noise
    
    def _generate_clutter(self, 
                         num_samples: int,
                         sample_rate: float,
                         environment: Environment) -> np.ndarray:
        """
        Generate clutter returns
        
        Args:
            num_samples: Number of samples
            sample_rate: Sample rate
            environment: Environmental conditions
            
        Returns:
            Complex clutter samples
        """
        # Simple clutter model - distributed scatterers
        num_clutter_patches = np.random.poisson(10)
        clutter = np.zeros(num_samples, dtype=complex)
        
        for _ in range(num_clutter_patches):
            # Random range
            clutter_range = np.random.uniform(100, 10000)
            delay_samples = int(2 * clutter_range / 3e8 * sample_rate)
            
            if delay_samples < num_samples:
                # Random RCS (log-normal distribution)
                clutter_rcs = np.random.lognormal(-2, 1)
                
                # Calculate clutter power
                clutter_power = self.radar.radar_equation(clutter_range, clutter_rcs)
                clutter_amplitude = np.sqrt(clutter_power)
                
                # Random Doppler (wind, movement)
                clutter_doppler = np.random.normal(0, 5)  # m/s
                doppler_shift = self.radar.doppler_shift(clutter_doppler)
                
                # Add to clutter signal
                clutter[delay_samples] += (
                    clutter_amplitude * 
                    np.exp(1j * 2 * np.pi * doppler_shift * delay_samples / sample_rate)
                )
        
        return clutter
    
    def _apply_iq_imbalance(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply I/Q imbalance
        
        Args:
            signal: Complex signal
            
        Returns:
            Signal with I/Q imbalance
        """
        i = signal.real
        q = signal.imag
        
        # Amplitude imbalance
        g = 1 + self.iq_params.iq_imbalance_amplitude
        
        # Phase imbalance
        phi = self.iq_params.iq_imbalance_phase
        
        # Apply imbalance
        i_out = i
        q_out = g * (q * np.cos(phi) + i * np.sin(phi))
        
        return i_out + 1j * q_out
    
    def _apply_phase_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply phase noise
        
        Args:
            signal: Complex signal
            
        Returns:
            Signal with phase noise
        """
        # Generate phase noise (simplified model)
        phase_noise_power = 10**(self.iq_params.phase_noise_level / 10)
        phase_noise = np.sqrt(phase_noise_power) * np.random.randn(len(signal))
        
        # Integrate to get phase walk
        phase_walk = np.cumsum(phase_noise) / self.waveform_gen.params.sample_rate
        
        # Apply phase noise
        return signal * np.exp(1j * phase_walk)
    
    def _quantize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Quantize signal (ADC effect)
        
        Args:
            signal: Complex signal
            
        Returns:
            Quantized signal
        """
        # Normalize to ADC range
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            scale = self.adc_scale / max_val
            signal_scaled = signal * scale
        else:
            signal_scaled = signal
        
        # Quantize I and Q separately
        i = signal_scaled.real
        q = signal_scaled.imag
        
        # Quantization levels
        levels = np.linspace(-self.adc_scale, self.adc_scale, self.adc_levels)
        
        # Quantize
        i_quantized = levels[np.argmin(np.abs(levels[:, np.newaxis] - i), axis=0)]
        q_quantized = levels[np.argmin(np.abs(levels[:, np.newaxis] - q), axis=0)]
        
        # Rescale back
        if max_val > 0:
            return (i_quantized + 1j * q_quantized) / scale
        else:
            return i_quantized + 1j * q_quantized
    
    def save_iq_data(self,
                    iq_data: np.ndarray,
                    filename: str,
                    format: str = "complex64",
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Save IQ data to file
        
        Args:
            iq_data: Complex IQ data
            filename: Output filename
            format: Data format ('complex64', 'int16', 'hdf5', 'mat')
            metadata: Optional metadata dictionary
        """
        if format == "complex64":
            # Save as interleaved float32 I/Q
            iq_interleaved = np.zeros(iq_data.size * 2, dtype=np.float32)
            iq_interleaved[0::2] = iq_data.real.flatten()
            iq_interleaved[1::2] = iq_data.imag.flatten()
            iq_interleaved.tofile(filename)
            
        elif format == "int16":
            # Save as interleaved int16 I/Q
            # Scale to int16 range
            scale = 32767 / np.max(np.abs(iq_data))
            i_int = (iq_data.real * scale).astype(np.int16)
            q_int = (iq_data.imag * scale).astype(np.int16)
            
            iq_interleaved = np.zeros(iq_data.size * 2, dtype=np.int16)
            iq_interleaved[0::2] = i_int.flatten()
            iq_interleaved[1::2] = q_int.flatten()
            iq_interleaved.tofile(filename)
            
        elif format == "hdf5":
            # Save as HDF5 with metadata
            with h5py.File(filename, 'w') as f:
                f.create_dataset('iq_data', data=iq_data)
                
                # Add metadata
                if metadata:
                    meta_group = f.create_group('metadata')
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str)):
                            meta_group.attrs[key] = value
                        else:
                            meta_group.attrs[key] = str(value)
                
                # Add generation parameters
                params_group = f.create_group('parameters')
                params_group.attrs['sample_rate'] = self.waveform_gen.params.sample_rate
                params_group.attrs['bandwidth'] = self.waveform_gen.params.bandwidth
                params_group.attrs['pulse_width'] = self.waveform_gen.params.pulse_width
                params_group.attrs['adc_bits'] = self.iq_params.adc_bits
                
        elif format == "mat":
            # Save as MATLAB .mat file
            try:
                from scipy.io import savemat
                mat_data = {
                    'iq_data': iq_data,
                    'sample_rate': self.waveform_gen.params.sample_rate,
                    'bandwidth': self.waveform_gen.params.bandwidth
                }
                if metadata:
                    mat_data.update(metadata)
                savemat(filename, mat_data)
            except ImportError:
                raise ImportError("scipy required for .mat file support")
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load_iq_data(self, filename: str, format: str = "complex64") -> Tuple[np.ndarray, Dict]:
        """
        Load IQ data from file
        
        Args:
            filename: Input filename
            format: Data format
            
        Returns:
            IQ data and metadata dictionary
        """
        metadata = {}
        
        if format == "complex64":
            # Load interleaved float32 I/Q
            iq_interleaved = np.fromfile(filename, dtype=np.float32)
            i = iq_interleaved[0::2]
            q = iq_interleaved[1::2]
            iq_data = i + 1j * q
            
        elif format == "int16":
            # Load interleaved int16 I/Q
            iq_interleaved = np.fromfile(filename, dtype=np.int16)
            i = iq_interleaved[0::2].astype(np.float32) / 32767
            q = iq_interleaved[1::2].astype(np.float32) / 32767
            iq_data = i + 1j * q
            
        elif format == "hdf5":
            # Load from HDF5
            with h5py.File(filename, 'r') as f:
                iq_data = f['iq_data'][:]
                
                # Load metadata
                if 'metadata' in f:
                    for key in f['metadata'].attrs:
                        metadata[key] = f['metadata'].attrs[key]
                
                # Load parameters
                if 'parameters' in f:
                    for key in f['parameters'].attrs:
                        metadata[key] = f['parameters'].attrs[key]
        
        elif format == "mat":
            # Load from MATLAB .mat file
            try:
                from scipy.io import loadmat
                mat_data = loadmat(filename)
                iq_data = mat_data['iq_data']
                
                # Extract metadata
                for key, value in mat_data.items():
                    if not key.startswith('__'):
                        metadata[key] = value
            except ImportError:
                raise ImportError("scipy required for .mat file support")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return iq_data, metadata
    
    def stream_iq_data(self,
                      targets: List[Target],
                      duration: float,
                      chunk_size: int = 1024,
                      waveform_type: WaveformType = WaveformType.LFM,
                      **waveform_kwargs):
        """
        Generator for streaming IQ data
        
        Args:
            targets: List of targets
            duration: Total duration in seconds
            chunk_size: Samples per chunk
            waveform_type: Type of waveform
            **waveform_kwargs: Additional waveform parameters
            
        Yields:
            Chunks of IQ data
        """
        sample_rate = self.waveform_gen.params.sample_rate
        total_samples = int(duration * sample_rate)
        
        # Generate transmit waveform
        tx_waveform = self.waveform_gen.generate(waveform_type, **waveform_kwargs)
        
        samples_generated = 0
        while samples_generated < total_samples:
            # Generate chunk
            current_chunk_size = min(chunk_size, total_samples - samples_generated)
            
            # For simplicity, generate noise for this chunk
            chunk = self._generate_noise(current_chunk_size)
            
            # Add target returns (simplified for streaming)
            for target in targets:
                target_range = target.motion.range
                delay_samples = int(2 * target_range / 3e8 * sample_rate)
                
                if samples_generated <= delay_samples < samples_generated + current_chunk_size:
                    # Target return starts in this chunk
                    rcs = target.get_rcs()
                    rx_power = self.radar.radar_equation(target_range, rcs)
                    amplitude = np.sqrt(rx_power)
                    
                    local_idx = delay_samples - samples_generated
                    chunk[local_idx:] += amplitude * tx_waveform[:current_chunk_size - local_idx]
            
            yield chunk
            samples_generated += current_chunk_size
    
    def analyze_iq_quality(self, iq_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze IQ data quality metrics
        
        Args:
            iq_data: Complex IQ data
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Signal statistics
        metrics['mean_power'] = np.mean(np.abs(iq_data)**2)
        metrics['peak_power'] = np.max(np.abs(iq_data)**2)
        metrics['dynamic_range'] = 10 * np.log10(
            metrics['peak_power'] / (metrics['mean_power'] + 1e-10)
        )
        
        # I/Q balance
        i = iq_data.real
        q = iq_data.imag
        metrics['iq_amplitude_balance'] = np.std(i) / np.std(q)
        metrics['iq_phase_orthogonality'] = np.corrcoef(i, q)[0, 1]
        
        # DC offset
        metrics['dc_offset_i'] = np.mean(i)
        metrics['dc_offset_q'] = np.mean(q)
        
        # Spectral properties
        spectrum = np.fft.fft(iq_data)
        power_spectrum = np.abs(spectrum)**2
        metrics['spectral_peak_freq'] = np.argmax(power_spectrum) * \
                                       self.waveform_gen.params.sample_rate / len(iq_data)
        
        return metrics