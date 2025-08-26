"""
Digital Radio Frequency Memory (DRFM) Jammer Implementation

This module implements a sophisticated DRFM jammer capable of:
- High-fidelity signal capture and replay
- Coherent false target generation
- Adaptive response to various radar waveforms
- Multiple simultaneous jamming techniques

Author: Claude Code
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
import warnings


class DRFMTechnique(Enum):
    """DRFM jamming techniques"""
    RANGE_GATE_PULL_OFF = "range_gate_pull_off"
    VELOCITY_GATE_PULL_OFF = "velocity_gate_pull_off"
    RANGE_GATE_PULL_IN = "range_gate_pull_in"
    ANGLE_GATE_PULL_OFF = "angle_gate_pull_off"
    FALSE_TARGET = "false_target"
    BARRAGE_JAMMING = "barrage_jamming"
    COVER_PULSE = "cover_pulse"
    INVERSE_GAIN = "inverse_gain"
    BLINKING = "blinking"
    DOPPLER_SHIFT = "doppler_shift"


@dataclass
class PulseDescriptorWord:
    """
    Pulse Descriptor Word (PDW) containing extracted pulse characteristics
    """
    # Timing parameters
    time_of_arrival: float = 0.0  # seconds
    pulse_width: float = 0.0  # seconds
    pulse_repetition_interval: float = 0.0  # seconds
    
    # Frequency parameters
    carrier_frequency: float = 0.0  # Hz
    bandwidth: float = 0.0  # Hz
    chirp_rate: float = 0.0  # Hz/s
    
    # Amplitude parameters
    amplitude: float = 0.0  # linear scale
    power: float = 0.0  # watts
    
    # Modulation parameters
    modulation_type: str = "unknown"  # "pulse", "lfm", "bpsk", "frank", etc.
    phase_code: Optional[np.ndarray] = None
    
    # Pulse shape
    envelope_type: str = "rectangular"  # "rectangular", "gaussian", "raised_cosine"
    
    # Signal quality metrics
    snr: float = 0.0  # dB
    confidence: float = 0.0  # 0-1 scale
    
    # Raw signal data
    iq_samples: Optional[np.ndarray] = field(default=None, repr=False)
    sample_rate: float = 0.0  # Hz


@dataclass
class FalseTarget:
    """
    Configuration for a false target
    """
    range_delay: float  # meters (additional delay from true target)
    doppler_shift: float  # Hz (Doppler frequency shift)
    amplitude_scale: float  # Linear amplitude scaling factor
    phase_offset: float = 0.0  # radians
    active: bool = True
    
    # Advanced parameters
    range_rate: float = 0.0  # m/s (for dynamic range changes)
    doppler_rate: float = 0.0  # Hz/s (for dynamic Doppler changes)
    amplitude_modulation: Optional[Dict[str, Any]] = None
    

class DRFMJammer:
    """
    Digital Radio Frequency Memory (DRFM) Jammer
    
    A sophisticated electronic warfare system capable of:
    - Capturing and storing radar signals with high fidelity
    - Analyzing pulse characteristics in real-time
    - Generating coherent false targets
    - Implementing various deception techniques
    """
    
    def __init__(self, 
                 sample_rate: float = 1e9,  # 1 GHz sampling
                 memory_depth: int = 1000000,  # 1M samples buffer
                 frequency_range: Tuple[float, float] = (1e9, 18e9),  # 1-18 GHz
                 dynamic_range: float = 80.0,  # dB
                 processing_delay: float = 100e-9):  # 100 ns
        """
        Initialize DRFM jammer
        
        Args:
            sample_rate: ADC/DAC sampling rate in Hz
            memory_depth: Number of complex samples that can be stored
            frequency_range: Operating frequency range (min_freq, max_freq) in Hz
            dynamic_range: System dynamic range in dB
            processing_delay: Minimum processing delay in seconds
        """
        self.sample_rate = sample_rate
        self.memory_depth = memory_depth
        self.frequency_range = frequency_range
        self.dynamic_range = dynamic_range
        self.processing_delay = processing_delay
        
        # Signal storage and analysis
        self.captured_signals: List[PulseDescriptorWord] = []
        self.signal_memory: Dict[str, np.ndarray] = {}
        self.pulse_library: Dict[str, PulseDescriptorWord] = {}
        
        # False target management
        self.false_targets: List[FalseTarget] = []
        self.max_false_targets = 16
        
        # System state
        self.is_active = True
        self.current_technique = DRFMTechnique.FALSE_TARGET
        self.coherent_processing = True
        self.phase_reference = 0.0
        
        # Performance metrics
        self.capture_efficiency = 0.95  # 95% successful captures
        self.replay_fidelity = 0.98  # 98% fidelity
        
        # Internal buffers
        self._input_buffer = np.zeros(memory_depth, dtype=complex)
        self._output_buffer = np.zeros(memory_depth, dtype=complex)
        self._analysis_buffer = np.zeros(memory_depth, dtype=complex)
        
        # Calibration and characterization
        self._phase_calibration = np.zeros(memory_depth, dtype=complex)
        self._amplitude_calibration = np.ones(memory_depth)
        
    def capture_signal(self, 
                      signal: np.ndarray, 
                      sampling_rate: Optional[float] = None,
                      trigger_threshold: float = -60.0) -> Optional[PulseDescriptorWord]:
        """
        Capture and store incoming radar pulse with high fidelity
        
        Args:
            signal: Complex IQ samples of received signal
            sampling_rate: Signal sampling rate (if different from system rate)
            trigger_threshold: Trigger threshold in dBm
            
        Returns:
            PulseDescriptorWord if pulse detected and captured, None otherwise
        """
        if sampling_rate is None:
            sampling_rate = self.sample_rate
            
        # Resample if necessary
        if sampling_rate != self.sample_rate:
            signal = self._resample_signal(signal, sampling_rate, self.sample_rate)
            
        # Detect pulse presence
        signal_power = np.abs(signal)**2
        power_db = 10 * np.log10(np.mean(signal_power) + 1e-12) + 30  # Convert to dBm
        
        if power_db < trigger_threshold:
            return None
            
        # Store in capture buffer
        capture_length = min(len(signal), self.memory_depth)
        self._input_buffer[:capture_length] = signal[:capture_length]
        
        # Analyze the pulse
        pdw = self.analyze_pulse(signal[:capture_length])
        
        if pdw.confidence > 0.5:  # Minimum confidence threshold
            # Store in pulse library
            pulse_id = f"pulse_{len(self.captured_signals):04d}"
            pdw.iq_samples = signal[:capture_length].copy()
            pdw.sample_rate = self.sample_rate
            
            self.captured_signals.append(pdw)
            self.pulse_library[pulse_id] = pdw
            
            # Maintain memory limits
            if len(self.captured_signals) > 1000:
                # Remove oldest captures
                old_pulse = self.captured_signals.pop(0)
                if old_pulse.iq_samples is not None:
                    # Find and remove from library
                    for key, value in list(self.pulse_library.items()):
                        if np.array_equal(value.iq_samples, old_pulse.iq_samples):
                            del self.pulse_library[key]
                            break
                            
        return pdw
    
    def analyze_pulse(self, signal: np.ndarray) -> PulseDescriptorWord:
        """
        Extract pulse characteristics and generate PDW
        
        Args:
            signal: Complex IQ samples
            
        Returns:
            PulseDescriptorWord with extracted characteristics
        """
        pdw = PulseDescriptorWord()
        
        # Basic signal parameters
        signal_length = len(signal)
        time_vector = np.arange(signal_length) / self.sample_rate
        
        # Amplitude analysis
        signal_magnitude = np.abs(signal)
        pdw.amplitude = np.max(signal_magnitude)
        pdw.power = np.mean(signal_magnitude**2)
        
        # Pulse width estimation (using -3dB points)
        peak_power = np.max(signal_magnitude**2)
        threshold = peak_power * 0.5  # -3dB
        above_threshold = signal_magnitude**2 > threshold
        
        if np.any(above_threshold):
            start_idx = np.where(above_threshold)[0][0]
            end_idx = np.where(above_threshold)[0][-1]
            pdw.pulse_width = (end_idx - start_idx) / self.sample_rate
        else:
            pdw.pulse_width = signal_length / self.sample_rate
            
        # Frequency analysis
        spectrum = fft(signal * np.hanning(len(signal)))
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        power_spectrum = np.abs(spectrum)**2
        
        # Center frequency estimation
        peak_freq_idx = np.argmax(power_spectrum)
        pdw.carrier_frequency = freqs[peak_freq_idx]
        
        # Bandwidth estimation (-3dB bandwidth)
        peak_power_spec = np.max(power_spectrum)
        bandwidth_threshold = peak_power_spec * 0.5
        above_bw_threshold = power_spectrum > bandwidth_threshold
        
        if np.any(above_bw_threshold):
            freq_indices = np.where(above_bw_threshold)[0]
            freq_span = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
            pdw.bandwidth = abs(freq_span)
        else:
            pdw.bandwidth = self.sample_rate / 10  # Default estimate
            
        # Modulation analysis
        pdw.modulation_type, pdw.chirp_rate = self._analyze_modulation(signal)
        
        # Phase code detection for pulse compression signals
        if pdw.modulation_type in ["bpsk", "frank", "p1", "p2", "p3", "p4"]:
            pdw.phase_code = self._extract_phase_code(signal)
            
        # SNR estimation
        pdw.snr = self._estimate_snr(signal)
        
        # Confidence based on analysis consistency
        pdw.confidence = self._calculate_confidence(signal, pdw)
        
        return pdw
    
    def generate_false_target(self, 
                            range_delay: float, 
                            doppler_shift: float = 0.0,
                            amplitude_scale: float = 1.0,
                            phase_offset: float = 0.0) -> FalseTarget:
        """
        Create a new false target configuration
        
        Args:
            range_delay: Additional range delay in meters
            doppler_shift: Doppler frequency shift in Hz
            amplitude_scale: Amplitude scaling factor (linear)
            phase_offset: Phase offset in radians
            
        Returns:
            FalseTarget configuration object
        """
        false_target = FalseTarget(
            range_delay=range_delay,
            doppler_shift=doppler_shift,
            amplitude_scale=amplitude_scale,
            phase_offset=phase_offset
        )
        
        if len(self.false_targets) < self.max_false_targets:
            self.false_targets.append(false_target)
        else:
            warnings.warn("Maximum number of false targets reached")
            
        return false_target
    
    def replay_signal(self, 
                     pulse_id: Optional[str] = None,
                     modifications: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Replay stored signal with optional modifications
        
        Args:
            pulse_id: ID of pulse to replay (if None, uses most recent)
            modifications: Dictionary of modifications to apply
            
        Returns:
            Complex IQ samples of modified replay signal
        """
        # Select pulse to replay
        if pulse_id is None:
            if not self.captured_signals:
                return np.zeros(1000, dtype=complex)
            pdw = self.captured_signals[-1]
        else:
            if pulse_id not in self.pulse_library:
                raise ValueError(f"Pulse ID {pulse_id} not found in library")
            pdw = self.pulse_library[pulse_id]
            
        if pdw.iq_samples is None:
            return np.zeros(1000, dtype=complex)
            
        # Start with original signal
        replay_signal = pdw.iq_samples.copy()
        
        # Apply modifications
        if modifications:
            replay_signal = self._apply_modifications(replay_signal, modifications)
            
        # Apply false targets
        for false_target in self.false_targets:
            if false_target.active:
                replay_signal = self._add_false_target(replay_signal, false_target, pdw)
                
        # Maintain phase coherence
        if self.coherent_processing:
            replay_signal = self._maintain_phase_coherence(replay_signal)
            
        return replay_signal
    
    def apply_drfm_processing(self, 
                            signal: np.ndarray,
                            technique: DRFMTechnique,
                            parameters: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply specific DRFM technique to signal
        
        Args:
            signal: Input signal to process
            technique: DRFM technique to apply
            parameters: Technique-specific parameters
            
        Returns:
            Processed signal with applied technique
        """
        if parameters is None:
            parameters = {}
            
        if technique == DRFMTechnique.RANGE_GATE_PULL_OFF:
            return self._range_gate_pull_off(signal, parameters)
        elif technique == DRFMTechnique.VELOCITY_GATE_PULL_OFF:
            return self._velocity_gate_pull_off(signal, parameters)
        elif technique == DRFMTechnique.RANGE_GATE_PULL_IN:
            return self._range_gate_pull_in(signal, parameters)
        elif technique == DRFMTechnique.FALSE_TARGET:
            return self._false_target_generation(signal, parameters)
        elif technique == DRFMTechnique.COVER_PULSE:
            return self._cover_pulse(signal, parameters)
        elif technique == DRFMTechnique.DOPPLER_SHIFT:
            return self._doppler_shift(signal, parameters)
        elif technique == DRFMTechnique.BLINKING:
            return self._blinking(signal, parameters)
        elif technique == DRFMTechnique.INVERSE_GAIN:
            return self._inverse_gain(signal, parameters)
        else:
            return signal  # Return unmodified signal for unknown techniques
    
    def add_false_target(self, false_target: FalseTarget) -> None:
        """
        Add a false target to the active list
        
        Args:
            false_target: FalseTarget configuration
        """
        if len(self.false_targets) < self.max_false_targets:
            self.false_targets.append(false_target)
        else:
            # Replace oldest false target
            self.false_targets.pop(0)
            self.false_targets.append(false_target)
    
    def remove_false_target(self, index: int) -> None:
        """
        Remove a false target by index
        
        Args:
            index: Index of false target to remove
        """
        if 0 <= index < len(self.false_targets):
            self.false_targets.pop(index)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'is_active': self.is_active,
            'current_technique': self.current_technique.value,
            'captured_pulses': len(self.captured_signals),
            'active_false_targets': len([ft for ft in self.false_targets if ft.active]),
            'memory_usage': len(self.captured_signals) / 1000 * 100,  # Percentage
            'sample_rate': self.sample_rate,
            'frequency_range': self.frequency_range,
            'processing_delay': self.processing_delay,
            'coherent_processing': self.coherent_processing
        }
    
    # Private helper methods
    
    def _resample_signal(self, signal: np.ndarray, 
                        input_rate: float, 
                        output_rate: float) -> np.ndarray:
        """Resample signal to match system sampling rate"""
        if input_rate == output_rate:
            return signal
            
        # Use scipy.signal.resample for high-quality resampling
        resample_ratio = output_rate / input_rate
        new_length = int(len(signal) * resample_ratio)
        
        return signal.resample(signal, new_length)
    
    def _analyze_modulation(self, signal: np.ndarray) -> Tuple[str, float]:
        """
        Analyze signal modulation type and parameters
        
        Returns:
            Tuple of (modulation_type, chirp_rate)
        """
        # Instantaneous phase analysis
        analytic_signal = signal
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Instantaneous frequency
        inst_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)
        
        # Check for linear frequency modulation (chirp)
        if len(inst_freq) > 10:
            # Linear regression to detect chirp
            time_indices = np.arange(len(inst_freq))
            chirp_coeff = np.polyfit(time_indices, inst_freq, 1)
            chirp_rate = chirp_coeff[0] * self.sample_rate
            
            # Check if frequency changes linearly
            linear_fit = np.polyval(chirp_coeff, time_indices)
            residual = np.std(inst_freq - linear_fit)
            
            if residual < np.std(inst_freq) * 0.1:  # Good linear fit
                return "lfm", chirp_rate
                
        # Check for phase-coded signals
        phase_diff = np.diff(phase)
        phase_transitions = np.where(np.abs(phase_diff) > np.pi/2)[0]
        
        if len(phase_transitions) > 0:
            # Potential phase-coded signal
            return "bpsk", 0.0
            
        # Default to pulse modulation
        return "pulse", 0.0
    
    def _extract_phase_code(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Extract phase code from coded signal"""
        # This is a simplified implementation
        # Real implementation would use sophisticated algorithms
        
        phase = np.unwrap(np.angle(signal))
        
        # Detect chip boundaries (simplified)
        # In practice, would use correlation with known sequences
        phase_transitions = np.where(np.abs(np.diff(phase)) > np.pi/2)[0]
        
        if len(phase_transitions) > 0:
            # Extract binary phase states
            return np.sign(np.cos(phase[::len(signal)//16]))  # Simplified
        
        return None
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Use signal power vs noise floor estimation
        signal_power = np.mean(np.abs(signal)**2)
        
        # Estimate noise from signal tails (simplified)
        tail_length = min(100, len(signal) // 10)
        noise_estimate = np.mean([
            np.mean(np.abs(signal[:tail_length])**2),
            np.mean(np.abs(signal[-tail_length:])**2)
        ])
        
        if noise_estimate > 0:
            snr_linear = signal_power / noise_estimate
            return 10 * np.log10(snr_linear)
        else:
            return 60.0  # High SNR default
    
    def _calculate_confidence(self, signal: np.ndarray, pdw: PulseDescriptorWord) -> float:
        """Calculate confidence in pulse analysis"""
        confidence_factors = []
        
        # SNR-based confidence
        if pdw.snr > 20:
            confidence_factors.append(1.0)
        elif pdw.snr > 10:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
            
        # Pulse width consistency
        if 1e-6 < pdw.pulse_width < 1e-3:  # Reasonable pulse width
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.6)
            
        # Bandwidth consistency
        if pdw.bandwidth > 0 and pdw.bandwidth < self.sample_rate / 2:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.7)
            
        return np.mean(confidence_factors)
    
    def _apply_modifications(self, signal: np.ndarray, 
                           modifications: Dict[str, Any]) -> np.ndarray:
        """Apply modifications to signal"""
        modified_signal = signal.copy()
        
        # Amplitude scaling
        if 'amplitude_scale' in modifications:
            modified_signal *= modifications['amplitude_scale']
            
        # Phase offset
        if 'phase_offset' in modifications:
            phase_shift = np.exp(1j * modifications['phase_offset'])
            modified_signal *= phase_shift
            
        # Frequency shift
        if 'frequency_shift' in modifications:
            freq_shift = modifications['frequency_shift']
            time_vector = np.arange(len(signal)) / self.sample_rate
            freq_shift_phasor = np.exp(1j * 2 * np.pi * freq_shift * time_vector)
            modified_signal *= freq_shift_phasor
            
        # Time delay (range delay)
        if 'time_delay' in modifications:
            delay_samples = int(modifications['time_delay'] * self.sample_rate)
            if delay_samples > 0:
                modified_signal = np.concatenate([
                    np.zeros(delay_samples, dtype=complex),
                    modified_signal
                ])[:len(signal)]
            elif delay_samples < 0:
                modified_signal = np.concatenate([
                    modified_signal[-delay_samples:],
                    np.zeros(-delay_samples, dtype=complex)
                ])[:len(signal)]
                
        return modified_signal
    
    def _add_false_target(self, signal: np.ndarray, 
                         false_target: FalseTarget,
                         pdw: PulseDescriptorWord) -> np.ndarray:
        """Add false target to signal"""
        # Calculate time delay from range delay
        c = 3e8  # Speed of light
        time_delay = 2 * false_target.range_delay / c  # Round-trip delay
        delay_samples = int(time_delay * self.sample_rate)
        
        # Create delayed and modified copy
        false_signal = signal.copy() * false_target.amplitude_scale
        
        # Apply Doppler shift
        if false_target.doppler_shift != 0:
            time_vector = np.arange(len(signal)) / self.sample_rate
            doppler_phasor = np.exp(1j * 2 * np.pi * false_target.doppler_shift * time_vector)
            false_signal *= doppler_phasor
            
        # Apply phase offset
        if false_target.phase_offset != 0:
            phase_phasor = np.exp(1j * false_target.phase_offset)
            false_signal *= phase_phasor
            
        # Add delayed false target
        combined_signal = signal.copy()
        if delay_samples > 0 and delay_samples < len(signal):
            combined_signal[delay_samples:] += false_signal[:-delay_samples]
        elif delay_samples < 0:
            combined_signal[:-delay_samples] += false_signal[-delay_samples:]
        else:
            combined_signal += false_signal
            
        return combined_signal
    
    def _maintain_phase_coherence(self, signal: np.ndarray) -> np.ndarray:
        """Maintain phase coherence across pulses"""
        # Apply phase reference correction
        if hasattr(self, 'phase_reference'):
            phase_correction = np.exp(-1j * self.phase_reference)
            return signal * phase_correction
        return signal
    
    # DRFM Technique Implementations
    
    def _range_gate_pull_off(self, signal: np.ndarray, 
                           parameters: Dict[str, Any]) -> np.ndarray:
        """Implement range gate pull-off technique"""
        pull_off_rate = parameters.get('pull_off_rate', 100.0)  # m/s
        duration = parameters.get('duration', 1e-3)  # seconds
        
        c = 3e8
        samples_per_second = self.sample_rate
        delay_rate_samples = 2 * pull_off_rate * samples_per_second / c
        
        output_signal = np.zeros_like(signal)
        for i in range(len(signal)):
            time = i / self.sample_rate
            if time < duration:
                delay_samples = int(delay_rate_samples * time)
                src_idx = i - delay_samples
                if 0 <= src_idx < len(signal):
                    output_signal[i] = signal[src_idx]
            else:
                output_signal[i] = signal[i]
                
        return output_signal
    
    def _velocity_gate_pull_off(self, signal: np.ndarray,
                              parameters: Dict[str, Any]) -> np.ndarray:
        """Implement velocity gate pull-off technique"""
        doppler_rate = parameters.get('doppler_rate', 1000.0)  # Hz/s
        duration = parameters.get('duration', 1e-3)  # seconds
        
        time_vector = np.arange(len(signal)) / self.sample_rate
        
        # Apply linearly increasing Doppler shift
        doppler_shift = np.minimum(doppler_rate * time_vector, 
                                  doppler_rate * duration)
        doppler_phasor = np.exp(1j * 2 * np.pi * doppler_shift * time_vector)
        
        return signal * doppler_phasor
    
    def _range_gate_pull_in(self, signal: np.ndarray,
                          parameters: Dict[str, Any]) -> np.ndarray:
        """Implement range gate pull-in technique"""
        pull_in_rate = parameters.get('pull_in_rate', -50.0)  # m/s (negative)
        return self._range_gate_pull_off(signal, {'pull_off_rate': pull_in_rate})
    
    def _false_target_generation(self, signal: np.ndarray,
                               parameters: Dict[str, Any]) -> np.ndarray:
        """Generate multiple false targets"""
        num_targets = parameters.get('num_targets', 3)
        range_offsets = parameters.get('range_offsets', [100, 200, 300])  # meters
        amplitude_scales = parameters.get('amplitude_scales', [0.8, 0.6, 0.4])
        
        combined_signal = signal.copy()
        
        for i in range(min(num_targets, len(range_offsets))):
            false_target = FalseTarget(
                range_delay=range_offsets[i],
                doppler_shift=0.0,
                amplitude_scale=amplitude_scales[i] if i < len(amplitude_scales) else 0.5
            )
            combined_signal = self._add_false_target(combined_signal, false_target, None)
            
        return combined_signal
    
    def _cover_pulse(self, signal: np.ndarray,
                    parameters: Dict[str, Any]) -> np.ndarray:
        """Generate cover pulse to mask true echo"""
        advance_time = parameters.get('advance_time', 1e-6)  # seconds
        amplitude_scale = parameters.get('amplitude_scale', 1.2)
        
        advance_samples = int(advance_time * self.sample_rate)
        
        # Create advanced copy of signal
        cover_signal = np.zeros_like(signal)
        if advance_samples < len(signal):
            cover_signal[:-advance_samples] = signal[advance_samples:] * amplitude_scale
            
        return cover_signal + signal
    
    def _doppler_shift(self, signal: np.ndarray,
                      parameters: Dict[str, Any]) -> np.ndarray:
        """Apply constant Doppler shift"""
        doppler_freq = parameters.get('doppler_frequency', 1000.0)  # Hz
        
        time_vector = np.arange(len(signal)) / self.sample_rate
        doppler_phasor = np.exp(1j * 2 * np.pi * doppler_freq * time_vector)
        
        return signal * doppler_phasor
    
    def _blinking(self, signal: np.ndarray,
                 parameters: Dict[str, Any]) -> np.ndarray:
        """Implement blinking (amplitude modulation)"""
        blink_frequency = parameters.get('blink_frequency', 100.0)  # Hz
        modulation_depth = parameters.get('modulation_depth', 0.8)
        
        time_vector = np.arange(len(signal)) / self.sample_rate
        amplitude_modulation = 1 + modulation_depth * np.sin(2 * np.pi * blink_frequency * time_vector)
        
        return signal * amplitude_modulation
    
    def _inverse_gain(self, signal: np.ndarray,
                     parameters: Dict[str, Any]) -> np.ndarray:
        """Implement inverse gain technique"""
        # Reduces signal amplitude proportionally to input strength
        gain_factor = parameters.get('gain_factor', -0.5)
        
        signal_power = np.abs(signal)**2
        max_power = np.max(signal_power)
        
        if max_power > 0:
            normalized_power = signal_power / max_power
            inverse_gain = 1 + gain_factor * normalized_power
            return signal * np.sqrt(inverse_gain)
        
        return signal