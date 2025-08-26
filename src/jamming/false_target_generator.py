"""
False Target Generator for Radar Electronic Warfare

This module implements sophisticated false target generation capabilities for electronic warfare
simulation. It provides realistic false targets with proper phase coherence, RCS modulation,
and coordinated pattern generation.

Key Features:
- Range and velocity false targets
- Coordinated swarm and formation patterns
- Realistic RCS fluctuation and micro-motion
- Phase-coherent signal processing
- Multiple pattern generation algorithms

Author: Claude Code
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
import warnings


class PatternType(Enum):
    """False target pattern types"""
    SWARM = "swarm"
    SCREEN = "screen"
    DECOY_CHAIN = "decoy_chain"
    FORMATION = "formation"
    RANDOM_WALK = "random_walk"
    SPIRAL = "spiral"
    EXPANDING = "expanding"
    CONTRACTING = "contracting"


class RCSModel(Enum):
    """RCS fluctuation models"""
    SWERLING_0 = "swerling_0"  # Constant RCS
    SWERLING_1 = "swerling_1"  # Rayleigh, slow fluctuation
    SWERLING_2 = "swerling_2"  # Rayleigh, fast fluctuation
    SWERLING_3 = "swerling_3"  # Chi-squared 4 DOF, slow
    SWERLING_4 = "swerling_4"  # Chi-squared 4 DOF, fast
    CUSTOM = "custom"


@dataclass
class FalseTargetParams:
    """Parameters for a single false target"""
    range_offset: float = 0.0  # Range offset in meters
    velocity_offset: float = 0.0  # Velocity offset in m/s
    amplitude_scale: float = 1.0  # Amplitude scaling factor
    phase_offset: float = 0.0  # Phase offset in radians
    rcs_mean: float = 1.0  # Mean RCS in m²
    rcs_std: float = 0.1  # RCS standard deviation
    doppler_spread: float = 0.0  # Additional Doppler spread in Hz
    micro_motion_freq: float = 0.0  # Micro-motion frequency in Hz
    micro_motion_amplitude: float = 0.0  # Micro-motion amplitude


@dataclass
class SwarmParams:
    """Parameters for swarm pattern generation"""
    num_targets: int = 10
    center_range: float = 1000.0  # meters
    center_velocity: float = 100.0  # m/s
    range_spread: float = 50.0  # meters
    velocity_spread: float = 10.0  # m/s
    formation_type: str = "random"  # random, line, v_formation, diamond
    coherence_factor: float = 0.8  # Inter-target coherence
    update_rate: float = 1.0  # Pattern update rate in Hz


@dataclass
class ScreenParams:
    """Parameters for screen pattern generation"""
    num_targets: int = 50
    range_start: float = 800.0  # meters
    range_end: float = 1200.0  # meters
    velocity_center: float = 0.0  # m/s
    velocity_spread: float = 20.0  # m/s
    density_profile: str = "uniform"  # uniform, gaussian, exponential
    amplitude_taper: float = 0.1  # Amplitude variation


@dataclass
class DecoyChainParams:
    """Parameters for decoy chain generation"""
    num_targets: int = 5
    initial_range: float = 1000.0  # meters
    range_increment: float = 100.0  # meters per target
    velocity_gradient: float = 5.0  # m/s velocity change per target
    activation_delay: float = 0.1  # seconds between activations
    lifetime: float = 5.0  # seconds each target stays active


@dataclass
class RCSProfile:
    """RCS fluctuation profile"""
    model: RCSModel = RCSModel.SWERLING_1
    mean_rcs: float = 1.0  # m²
    std_rcs: float = 0.5  # m²
    correlation_time: float = 0.1  # seconds
    custom_profile: Optional[Callable[[float], float]] = None


class FalseTargetGenerator:
    """
    Advanced false target generator for electronic warfare simulation
    
    This class provides comprehensive false target generation capabilities including
    range/velocity false targets, coordinated patterns, and realistic target characteristics.
    """
    
    def __init__(self, 
                 sample_rate: float,
                 carrier_frequency: float,
                 speed_of_light: float = 3e8):
        """
        Initialize the false target generator
        
        Args:
            sample_rate: Sample rate in Hz
            carrier_frequency: Carrier frequency in Hz
            speed_of_light: Speed of light in m/s
        """
        self.sample_rate = sample_rate
        self.carrier_frequency = carrier_frequency
        self.speed_of_light = speed_of_light
        
        # Internal state for coherent processing
        self._pulse_count = 0
        self._phase_history = {}
        self._rcs_history = {}
        self._pattern_state = {}
        
        # Wavelength for Doppler calculations
        self.wavelength = speed_of_light / carrier_frequency
        
    def generate_range_false_targets(self,
                                   signal: np.ndarray,
                                   range_offsets: List[float],
                                   amplitudes: List[float],
                                   phases: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate false targets at specified range offsets
        
        Args:
            signal: Input radar signal
            range_offsets: List of range offsets in meters
            amplitudes: List of amplitude scaling factors
            phases: Optional list of phase offsets in radians
            
        Returns:
            Combined signal with false targets
        """
        if phases is None:
            phases = [0.0] * len(range_offsets)
            
        if len(range_offsets) != len(amplitudes) or len(range_offsets) != len(phases):
            raise ValueError("Range offsets, amplitudes, and phases must have same length")
            
        # Calculate time delays
        time_delays = [2 * r / self.speed_of_light for r in range_offsets]
        
        # Generate composite signal
        composite_signal = signal.copy()
        
        for delay, amp, phase in zip(time_delays, amplitudes, phases):
            # Calculate sample delay
            sample_delay = int(delay * self.sample_rate)
            
            if sample_delay < len(signal):
                # Create delayed and scaled replica
                delayed_signal = np.zeros_like(signal, dtype=complex)
                delayed_signal[sample_delay:] = signal[:-sample_delay] if sample_delay > 0 else signal
                
                # Apply amplitude and phase scaling
                delayed_signal *= amp * np.exp(1j * phase)
                
                # Add to composite
                composite_signal += delayed_signal
                
        return composite_signal
    
    def generate_velocity_false_targets(self,
                                      signal: np.ndarray,
                                      velocity_offsets: List[float],
                                      coherent_integration_time: float,
                                      amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate false targets with Doppler shifts
        
        Args:
            signal: Input radar signal
            velocity_offsets: List of velocity offsets in m/s
            coherent_integration_time: Coherent integration time in seconds
            amplitudes: Optional amplitude scaling factors
            
        Returns:
            Signal with Doppler-shifted false targets
        """
        if amplitudes is None:
            amplitudes = [1.0] * len(velocity_offsets)
            
        # Calculate Doppler frequencies
        doppler_freqs = [2 * v / self.wavelength for v in velocity_offsets]
        
        # Time vector
        t = np.arange(len(signal)) / self.sample_rate
        
        # Generate composite signal
        composite_signal = signal.copy()
        
        for doppler_freq, amp in zip(doppler_freqs, amplitudes):
            # Apply Doppler shift
            doppler_phase = 2 * np.pi * doppler_freq * t
            shifted_signal = signal * np.exp(1j * doppler_phase) * amp
            
            # Add to composite
            composite_signal += shifted_signal
            
        return composite_signal
    
    def generate_swarm_pattern(self,
                             signal: np.ndarray,
                             swarm_params: SwarmParams) -> Tuple[np.ndarray, List[FalseTargetParams]]:
        """
        Generate a coordinated swarm of false targets
        
        Args:
            signal: Input radar signal
            swarm_params: Swarm configuration parameters
            
        Returns:
            Tuple of (composite signal, list of target parameters)
        """
        targets = []
        
        # Generate target positions based on formation type
        if swarm_params.formation_type == "random":
            ranges = np.random.normal(swarm_params.center_range, 
                                    swarm_params.range_spread, 
                                    swarm_params.num_targets)
            velocities = np.random.normal(swarm_params.center_velocity,
                                        swarm_params.velocity_spread,
                                        swarm_params.num_targets)
        elif swarm_params.formation_type == "line":
            ranges = np.linspace(swarm_params.center_range - swarm_params.range_spread,
                               swarm_params.center_range + swarm_params.range_spread,
                               swarm_params.num_targets)
            velocities = np.full(swarm_params.num_targets, swarm_params.center_velocity)
        elif swarm_params.formation_type == "v_formation":
            ranges, velocities = self._generate_v_formation(swarm_params)
        elif swarm_params.formation_type == "diamond":
            ranges, velocities = self._generate_diamond_formation(swarm_params)
        else:
            raise ValueError(f"Unknown formation type: {swarm_params.formation_type}")
        
        # Create target parameters
        for i, (r, v) in enumerate(zip(ranges, velocities)):
            target = FalseTargetParams(
                range_offset=r - swarm_params.center_range,
                velocity_offset=v - swarm_params.center_velocity,
                amplitude_scale=0.5 + 0.5 * np.random.random(),  # Random amplitude variation
                phase_offset=2 * np.pi * np.random.random(),
                rcs_mean=1.0,
                rcs_std=0.2
            )
            targets.append(target)
        
        # Generate composite signal
        composite_signal = self._apply_target_list(signal, targets)
        
        return composite_signal, targets
    
    def generate_screen_pattern(self,
                              signal: np.ndarray,
                              screen_params: ScreenParams) -> Tuple[np.ndarray, List[FalseTargetParams]]:
        """
        Generate a screen (wall) of false targets
        
        Args:
            signal: Input radar signal
            screen_params: Screen configuration parameters
            
        Returns:
            Tuple of (composite signal, list of target parameters)
        """
        targets = []
        
        # Generate range distribution
        if screen_params.density_profile == "uniform":
            ranges = np.linspace(screen_params.range_start,
                               screen_params.range_end,
                               screen_params.num_targets)
        elif screen_params.density_profile == "gaussian":
            center = (screen_params.range_start + screen_params.range_end) / 2
            spread = (screen_params.range_end - screen_params.range_start) / 6
            ranges = np.random.normal(center, spread, screen_params.num_targets)
            ranges = np.clip(ranges, screen_params.range_start, screen_params.range_end)
        elif screen_params.density_profile == "exponential":
            ranges = np.random.exponential(
                (screen_params.range_end - screen_params.range_start) / 3,
                screen_params.num_targets
            ) + screen_params.range_start
            ranges = np.clip(ranges, screen_params.range_start, screen_params.range_end)
        else:
            raise ValueError(f"Unknown density profile: {screen_params.density_profile}")
        
        # Generate velocities
        velocities = np.random.normal(screen_params.velocity_center,
                                    screen_params.velocity_spread,
                                    screen_params.num_targets)
        
        # Create target parameters with amplitude tapering
        for i, (r, v) in enumerate(zip(ranges, velocities)):
            # Amplitude tapering based on position
            normalized_pos = (r - screen_params.range_start) / (screen_params.range_end - screen_params.range_start)
            taper_factor = 1.0 - screen_params.amplitude_taper * abs(normalized_pos - 0.5) * 2
            
            target = FalseTargetParams(
                range_offset=r - (screen_params.range_start + screen_params.range_end) / 2,
                velocity_offset=v,
                amplitude_scale=taper_factor * (0.8 + 0.4 * np.random.random()),
                phase_offset=2 * np.pi * np.random.random()
            )
            targets.append(target)
        
        # Generate composite signal
        composite_signal = self._apply_target_list(signal, targets)
        
        return composite_signal, targets
    
    def generate_decoy_chain(self,
                           signal: np.ndarray,
                           chain_params: DecoyChainParams,
                           current_time: float) -> Tuple[np.ndarray, List[FalseTargetParams]]:
        """
        Generate a sequential chain of decoy targets
        
        Args:
            signal: Input radar signal
            chain_params: Chain configuration parameters
            current_time: Current simulation time in seconds
            
        Returns:
            Tuple of (composite signal, list of active target parameters)
        """
        targets = []
        
        for i in range(chain_params.num_targets):
            # Calculate activation time for this target
            activation_time = i * chain_params.activation_delay
            
            # Check if target should be active
            if (current_time >= activation_time and 
                current_time < activation_time + chain_params.lifetime):
                
                range_offset = i * chain_params.range_increment
                velocity_offset = i * chain_params.velocity_gradient
                
                # Age-based amplitude modulation
                age = current_time - activation_time
                age_factor = 1.0 - (age / chain_params.lifetime) * 0.3  # Gradual fading
                
                target = FalseTargetParams(
                    range_offset=range_offset,
                    velocity_offset=velocity_offset,
                    amplitude_scale=age_factor,
                    phase_offset=2 * np.pi * np.random.random()
                )
                targets.append(target)
        
        # Generate composite signal
        composite_signal = self._apply_target_list(signal, targets)
        
        return composite_signal, targets
    
    def apply_rcs_modulation(self,
                           signal: np.ndarray,
                           rcs_profile: RCSProfile,
                           target_id: str = "default") -> np.ndarray:
        """
        Apply realistic RCS fluctuation to signal
        
        Args:
            signal: Input signal
            rcs_profile: RCS fluctuation parameters
            target_id: Unique identifier for target (for history tracking)
            
        Returns:
            Signal with RCS modulation applied
        """
        if rcs_profile.model == RCSModel.SWERLING_0:
            # Constant RCS
            scale_factor = np.sqrt(rcs_profile.mean_rcs)
            
        elif rcs_profile.model == RCSModel.SWERLING_1:
            # Rayleigh distribution, slow fluctuation
            if target_id not in self._rcs_history:
                self._rcs_history[target_id] = {
                    'last_update': 0,
                    'current_rcs': rcs_profile.mean_rcs
                }
            
            # Update RCS if correlation time has passed
            if (self._pulse_count - self._rcs_history[target_id]['last_update']) * \
               (1/self.sample_rate) > rcs_profile.correlation_time:
                self._rcs_history[target_id]['current_rcs'] = np.random.exponential(rcs_profile.mean_rcs)
                self._rcs_history[target_id]['last_update'] = self._pulse_count
            
            scale_factor = np.sqrt(self._rcs_history[target_id]['current_rcs'])
            
        elif rcs_profile.model == RCSModel.SWERLING_2:
            # Rayleigh distribution, fast fluctuation
            rcs_value = np.random.exponential(rcs_profile.mean_rcs)
            scale_factor = np.sqrt(rcs_value)
            
        elif rcs_profile.model == RCSModel.SWERLING_3:
            # Chi-squared 4 DOF, slow fluctuation
            if target_id not in self._rcs_history:
                self._rcs_history[target_id] = {
                    'last_update': 0,
                    'current_rcs': rcs_profile.mean_rcs
                }
            
            if (self._pulse_count - self._rcs_history[target_id]['last_update']) * \
               (1/self.sample_rate) > rcs_profile.correlation_time:
                # Chi-squared with 4 degrees of freedom
                rcs_value = np.random.gamma(2, rcs_profile.mean_rcs / 2)
                self._rcs_history[target_id]['current_rcs'] = rcs_value
                self._rcs_history[target_id]['last_update'] = self._pulse_count
            
            scale_factor = np.sqrt(self._rcs_history[target_id]['current_rcs'])
            
        elif rcs_profile.model == RCSModel.SWERLING_4:
            # Chi-squared 4 DOF, fast fluctuation
            rcs_value = np.random.gamma(2, rcs_profile.mean_rcs / 2)
            scale_factor = np.sqrt(rcs_value)
            
        elif rcs_profile.model == RCSModel.CUSTOM:
            if rcs_profile.custom_profile is None:
                raise ValueError("Custom RCS profile function not provided")
            current_time = self._pulse_count / self.sample_rate
            rcs_value = rcs_profile.custom_profile(current_time)
            scale_factor = np.sqrt(rcs_value)
            
        else:
            raise ValueError(f"Unknown RCS model: {rcs_profile.model}")
        
        return signal * scale_factor
    
    def generate_formation_pattern(self,
                                 signal: np.ndarray,
                                 pattern_type: PatternType,
                                 num_targets: int,
                                 center_range: float,
                                 center_velocity: float,
                                 scale_factor: float = 100.0) -> Tuple[np.ndarray, List[FalseTargetParams]]:
        """
        Generate specific formation patterns
        
        Args:
            signal: Input radar signal
            pattern_type: Type of pattern to generate
            num_targets: Number of targets in formation
            center_range: Center range in meters
            center_velocity: Center velocity in m/s
            scale_factor: Scaling factor for pattern size
            
        Returns:
            Tuple of (composite signal, list of target parameters)
        """
        if pattern_type == PatternType.SPIRAL:
            return self._generate_spiral_pattern(signal, num_targets, center_range, 
                                               center_velocity, scale_factor)
        elif pattern_type == PatternType.EXPANDING:
            return self._generate_expanding_pattern(signal, num_targets, center_range,
                                                  center_velocity, scale_factor)
        elif pattern_type == PatternType.CONTRACTING:
            return self._generate_contracting_pattern(signal, num_targets, center_range,
                                                    center_velocity, scale_factor)
        elif pattern_type == PatternType.RANDOM_WALK:
            return self._generate_random_walk_pattern(signal, num_targets, center_range,
                                                    center_velocity, scale_factor)
        else:
            raise ValueError(f"Pattern type {pattern_type} not implemented in this method")
    
    def add_micro_motion(self,
                        signal: np.ndarray,
                        micro_motion_freq: float,
                        micro_motion_amplitude: float,
                        phase_offset: float = 0.0) -> np.ndarray:
        """
        Add micro-motion signatures to signal for realism
        
        Args:
            signal: Input signal
            micro_motion_freq: Micro-motion frequency in Hz
            micro_motion_amplitude: Micro-motion amplitude
            phase_offset: Phase offset for micro-motion
            
        Returns:
            Signal with micro-motion applied
        """
        t = np.arange(len(signal)) / self.sample_rate
        
        # Generate micro-motion phase modulation
        micro_motion_phase = micro_motion_amplitude * np.sin(
            2 * np.pi * micro_motion_freq * t + phase_offset
        )
        
        # Apply micro-motion
        modulated_signal = signal * np.exp(1j * micro_motion_phase)
        
        return modulated_signal
    
    def update_pulse_count(self):
        """Update internal pulse counter for coherent processing"""
        self._pulse_count += 1
    
    def reset_coherent_state(self):
        """Reset all coherent processing state"""
        self._pulse_count = 0
        self._phase_history.clear()
        self._rcs_history.clear()
        self._pattern_state.clear()
    
    # Private helper methods
    def _apply_target_list(self, signal: np.ndarray, targets: List[FalseTargetParams]) -> np.ndarray:
        """Apply a list of targets to the signal"""
        composite_signal = signal.copy()
        
        for target in targets:
            # Range delay
            time_delay = 2 * target.range_offset / self.speed_of_light
            sample_delay = int(time_delay * self.sample_rate)
            
            # Velocity Doppler
            doppler_freq = 2 * target.velocity_offset / self.wavelength
            t = np.arange(len(signal)) / self.sample_rate
            doppler_phase = 2 * np.pi * doppler_freq * t
            
            # Create target signal
            target_signal = signal.copy()
            
            # Apply delay
            if sample_delay > 0 and sample_delay < len(signal):
                delayed_signal = np.zeros_like(signal, dtype=complex)
                delayed_signal[sample_delay:] = signal[:-sample_delay]
                target_signal = delayed_signal
            
            # Apply Doppler, amplitude, and phase
            target_signal *= (target.amplitude_scale * 
                            np.exp(1j * (doppler_phase + target.phase_offset)))
            
            # Add micro-motion if specified
            if target.micro_motion_freq > 0:
                target_signal = self.add_micro_motion(
                    target_signal,
                    target.micro_motion_freq,
                    target.micro_motion_amplitude
                )
            
            # Add Doppler spread if specified
            if target.doppler_spread > 0:
                spread_phase = np.random.normal(0, target.doppler_spread * 2 * np.pi, len(t))
                target_signal *= np.exp(1j * np.cumsum(spread_phase) / self.sample_rate)
            
            composite_signal += target_signal
        
        return composite_signal
    
    def _generate_v_formation(self, params: SwarmParams) -> Tuple[np.ndarray, np.ndarray]:
        """Generate V-formation pattern"""
        ranges = []
        velocities = []
        
        half_targets = params.num_targets // 2
        
        # Left wing
        for i in range(half_targets):
            offset = (i + 1) * params.range_spread / half_targets
            ranges.append(params.center_range - offset)
            velocities.append(params.center_velocity - offset * 0.1)
        
        # Leader
        if params.num_targets % 2 == 1:
            ranges.append(params.center_range)
            velocities.append(params.center_velocity)
        
        # Right wing
        for i in range(half_targets):
            offset = (i + 1) * params.range_spread / half_targets
            ranges.append(params.center_range - offset)
            velocities.append(params.center_velocity + offset * 0.1)
        
        return np.array(ranges), np.array(velocities)
    
    def _generate_diamond_formation(self, params: SwarmParams) -> Tuple[np.ndarray, np.ndarray]:
        """Generate diamond formation pattern"""
        ranges = []
        velocities = []
        
        quarter = params.num_targets // 4
        
        # Front point
        ranges.append(params.center_range + params.range_spread)
        velocities.append(params.center_velocity)
        
        # Sides
        for i in range(1, quarter + 1):
            # Left side
            ranges.append(params.center_range)
            velocities.append(params.center_velocity - i * params.velocity_spread / quarter)
            
            # Right side
            ranges.append(params.center_range)
            velocities.append(params.center_velocity + i * params.velocity_spread / quarter)
        
        # Rear point
        ranges.append(params.center_range - params.range_spread)
        velocities.append(params.center_velocity)
        
        # Fill remaining targets randomly
        while len(ranges) < params.num_targets:
            ranges.append(params.center_range + np.random.uniform(-params.range_spread/2, params.range_spread/2))
            velocities.append(params.center_velocity + np.random.uniform(-params.velocity_spread/2, params.velocity_spread/2))
        
        return np.array(ranges[:params.num_targets]), np.array(velocities[:params.num_targets])
    
    def _generate_spiral_pattern(self, signal, num_targets, center_range, center_velocity, scale_factor):
        """Generate spiral pattern"""
        targets = []
        angles = np.linspace(0, 4*np.pi, num_targets)
        radii = np.linspace(0, scale_factor, num_targets)
        
        for angle, radius in zip(angles, radii):
            range_offset = radius * np.cos(angle)
            velocity_offset = radius * np.sin(angle) * 0.1  # Scale velocity component
            
            target = FalseTargetParams(
                range_offset=range_offset,
                velocity_offset=velocity_offset,
                amplitude_scale=0.8 + 0.4 * np.random.random()
            )
            targets.append(target)
        
        composite_signal = self._apply_target_list(signal, targets)
        return composite_signal, targets
    
    def _generate_expanding_pattern(self, signal, num_targets, center_range, center_velocity, scale_factor):
        """Generate expanding pattern"""
        targets = []
        angles = np.linspace(0, 2*np.pi, num_targets, endpoint=False)
        
        # Use pattern state to track expansion
        if 'expanding_radius' not in self._pattern_state:
            self._pattern_state['expanding_radius'] = scale_factor * 0.1
        
        radius = self._pattern_state['expanding_radius']
        self._pattern_state['expanding_radius'] += scale_factor * 0.01  # Gradual expansion
        
        for angle in angles:
            range_offset = radius * np.cos(angle)
            velocity_offset = radius * np.sin(angle) * 0.05
            
            target = FalseTargetParams(
                range_offset=range_offset,
                velocity_offset=velocity_offset,
                amplitude_scale=1.0 / (1.0 + radius / scale_factor)  # Fade with distance
            )
            targets.append(target)
        
        composite_signal = self._apply_target_list(signal, targets)
        return composite_signal, targets
    
    def _generate_contracting_pattern(self, signal, num_targets, center_range, center_velocity, scale_factor):
        """Generate contracting pattern"""
        targets = []
        angles = np.linspace(0, 2*np.pi, num_targets, endpoint=False)
        
        # Use pattern state to track contraction
        if 'contracting_radius' not in self._pattern_state:
            self._pattern_state['contracting_radius'] = scale_factor
        
        radius = self._pattern_state['contracting_radius']
        self._pattern_state['contracting_radius'] = max(scale_factor * 0.1, radius - scale_factor * 0.02)
        
        for angle in angles:
            range_offset = radius * np.cos(angle)
            velocity_offset = radius * np.sin(angle) * 0.05
            
            target = FalseTargetParams(
                range_offset=range_offset,
                velocity_offset=velocity_offset,
                amplitude_scale=1.0 + (scale_factor - radius) / scale_factor  # Brighten as they contract
            )
            targets.append(target)
        
        composite_signal = self._apply_target_list(signal, targets)
        return composite_signal, targets
    
    def _generate_random_walk_pattern(self, signal, num_targets, center_range, center_velocity, scale_factor):
        """Generate random walk pattern"""
        targets = []
        
        # Initialize or update random walk state
        if 'random_walk_positions' not in self._pattern_state:
            self._pattern_state['random_walk_positions'] = [
                {'range': 0.0, 'velocity': 0.0} for _ in range(num_targets)
            ]
        
        positions = self._pattern_state['random_walk_positions']
        
        # Update positions with random walk
        for i, pos in enumerate(positions[:num_targets]):
            # Random walk step
            range_step = np.random.normal(0, scale_factor * 0.01)
            velocity_step = np.random.normal(0, scale_factor * 0.001)
            
            pos['range'] += range_step
            pos['velocity'] += velocity_step
            
            # Boundary conditions
            pos['range'] = np.clip(pos['range'], -scale_factor, scale_factor)
            pos['velocity'] = np.clip(pos['velocity'], -scale_factor * 0.1, scale_factor * 0.1)
            
            target = FalseTargetParams(
                range_offset=pos['range'],
                velocity_offset=pos['velocity'],
                amplitude_scale=0.7 + 0.6 * np.random.random()
            )
            targets.append(target)
        
        composite_signal = self._apply_target_list(signal, targets)
        return composite_signal, targets


# Utility functions for pattern generation
def create_coordinated_trajectory(num_targets: int, 
                                time_duration: float, 
                                pattern_type: PatternType,
                                **kwargs) -> Dict[str, np.ndarray]:
    """
    Create coordinated trajectories for multiple false targets
    
    Args:
        num_targets: Number of targets
        time_duration: Duration of trajectory in seconds
        pattern_type: Type of coordinated pattern
        **kwargs: Additional parameters for specific patterns
        
    Returns:
        Dictionary with 'ranges' and 'velocities' arrays of shape (num_targets, time_steps)
    """
    time_steps = int(time_duration * kwargs.get('update_rate', 10))
    t = np.linspace(0, time_duration, time_steps)
    
    ranges = np.zeros((num_targets, time_steps))
    velocities = np.zeros((num_targets, time_steps))
    
    if pattern_type == PatternType.FORMATION:
        # Maintain formation while moving
        base_range = kwargs.get('base_range', 1000.0)
        base_velocity = kwargs.get('base_velocity', 100.0)
        formation_spacing = kwargs.get('formation_spacing', 50.0)
        
        for i in range(num_targets):
            ranges[i, :] = base_range + i * formation_spacing
            velocities[i, :] = base_velocity + 10 * np.sin(2 * np.pi * 0.1 * t)  # Slight oscillation
            
    elif pattern_type == PatternType.SPIRAL:
        # Spiral trajectory
        center_range = kwargs.get('center_range', 1000.0)
        spiral_rate = kwargs.get('spiral_rate', 0.5)
        radius_rate = kwargs.get('radius_rate', 50.0)
        
        for i in range(num_targets):
            angle_offset = 2 * np.pi * i / num_targets
            angles = 2 * np.pi * spiral_rate * t + angle_offset
            radii = radius_rate * t / time_duration
            
            ranges[i, :] = center_range + radii * np.cos(angles)
            velocities[i, :] = radii * np.sin(angles) * 0.1
            
    # Add more pattern types as needed
    
    return {'ranges': ranges, 'velocities': velocities}


def calculate_false_target_effectiveness(original_signal: np.ndarray,
                                       false_target_signal: np.ndarray,
                                       detection_threshold: float) -> Dict[str, float]:
    """
    Calculate effectiveness metrics for false target generation
    
    Args:
        original_signal: Original radar signal
        false_target_signal: Signal with false targets
        detection_threshold: Detection threshold
        
    Returns:
        Dictionary with effectiveness metrics
    """
    # Signal-to-noise ratio analysis
    original_power = np.mean(np.abs(original_signal)**2)
    false_target_power = np.mean(np.abs(false_target_signal - original_signal)**2)
    
    # False target to signal ratio
    ft_to_signal_ratio = false_target_power / original_power
    
    # Detection probability estimation (simplified)
    signal_peaks = np.abs(original_signal)
    false_peaks = np.abs(false_target_signal - original_signal)
    
    detectable_false_targets = np.sum(false_peaks > detection_threshold)
    total_false_targets = len(false_peaks)
    
    detection_probability = detectable_false_targets / total_false_targets if total_false_targets > 0 else 0
    
    return {
        'false_target_to_signal_ratio_db': 10 * np.log10(ft_to_signal_ratio),
        'detection_probability': detection_probability,
        'total_false_targets': total_false_targets,
        'detectable_false_targets': detectable_false_targets
    }