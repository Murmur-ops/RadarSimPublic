"""
Gate Pull-Off Implementation for Radar Jamming

This module implements Range Gate Pull-Off (RGPO) and Velocity Gate Pull-Off (VGPO) 
techniques as sophisticated radar deception methods. These techniques aim to break 
radar track locks by gradually or abruptly shifting the apparent position or 
velocity of targets.

Key Features:
- Range Gate Pull-Off (RGPO) with multiple profiles
- Velocity Gate Pull-Off (VGPO) with adaptive rates
- Combined range-velocity pull-off
- Advanced pull-off profiles (linear, exponential, S-curve, step, random walk)
- Adaptive rate calculation based on radar parameters
- Smooth transition maintenance for skin return preservation
- Multi-target coordinated pull-off
- Gate position feedback monitoring

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
import time


class PullOffDirection(Enum):
    """Pull-off direction options"""
    PULL_AHEAD = "pull_ahead"      # Increase apparent range/velocity
    PULL_BACK = "pull_back"        # Decrease apparent range/velocity
    PULL_UP = "pull_up"            # Increase velocity (towards radar)
    PULL_DOWN = "pull_down"        # Decrease velocity (away from radar)
    BIDIRECTIONAL = "bidirectional" # Alternate directions


class PullOffProfile(Enum):
    """Pull-off rate profiles"""
    LINEAR = "linear"              # Constant pull rate
    EXPONENTIAL = "exponential"    # Accelerating pull rate
    S_CURVE = "s_curve"           # Smooth acceleration/deceleration
    STEP = "step"                 # Discrete jumps
    RANDOM_WALK = "random_walk"   # Unpredictable variations
    ADAPTIVE = "adaptive"         # Radar-responsive rate changes


class GateStatus(Enum):
    """Gate tracking status"""
    LOCKED = "locked"             # Gate is tracking
    BREAKING = "breaking"         # Gate lock degrading
    BROKEN = "broken"             # Gate lock lost
    RECOVERING = "recovering"     # Attempting to reacquire
    UNKNOWN = "unknown"           # Status uncertain


@dataclass
class RangeParams:
    """Range gate pull-off parameters"""
    initial_delay: float = 0.0         # Initial range delay in meters
    pull_rate: float = 100.0           # Pull rate in m/s
    direction: PullOffDirection = PullOffDirection.PULL_AHEAD
    profile: PullOffProfile = PullOffProfile.LINEAR
    max_delay: float = 1000.0          # Maximum range delay in meters
    min_delay: float = -500.0          # Minimum range delay in meters
    duration: float = 1.0              # Pull-off duration in seconds
    
    # Advanced parameters
    acceleration: float = 0.0          # m/s² for exponential profiles
    jitter_amplitude: float = 0.0      # Random variation amplitude (meters)
    step_size: float = 50.0           # Step size for step profile (meters)
    step_interval: float = 0.1        # Step interval for step profile (seconds)


@dataclass
class VelocityParams:
    """Velocity gate pull-off parameters"""
    initial_doppler: float = 0.0       # Initial Doppler shift in Hz
    pull_rate: float = 1000.0          # Pull rate in Hz/s
    direction: PullOffDirection = PullOffDirection.PULL_UP
    profile: PullOffProfile = PullOffProfile.LINEAR
    max_doppler: float = 10000.0       # Maximum Doppler shift in Hz
    min_doppler: float = -10000.0      # Minimum Doppler shift in Hz
    duration: float = 1.0              # Pull-off duration in seconds
    
    # Advanced parameters
    acceleration: float = 0.0          # Hz/s² for exponential profiles
    jitter_amplitude: float = 0.0      # Random variation amplitude (Hz)
    step_size: float = 500.0          # Step size for step profile (Hz)
    step_interval: float = 0.1        # Step interval for step profile (seconds)


@dataclass
class TransitionParams:
    """Skin return transition parameters"""
    transition_duration: float = 0.1   # Transition time in seconds
    overlap_factor: float = 0.5        # Overlap between skin and false returns
    amplitude_taper: str = "hanning"   # Tapering window type
    phase_continuity: bool = True      # Maintain phase continuity
    power_ratio_threshold: float = 0.1 # Minimum power ratio to maintain


@dataclass
class RadarParams:
    """Radar system parameters for adaptive calculations"""
    prf: float = 1000.0                # Pulse repetition frequency in Hz
    pulse_width: float = 1e-6          # Pulse width in seconds
    bandwidth: float = 10e6            # Signal bandwidth in Hz
    carrier_frequency: float = 10e9    # Carrier frequency in Hz
    range_gate_width: float = 150.0    # Range gate width in meters
    velocity_gate_width: float = 100.0 # Velocity gate width in Hz
    agc_time_constant: float = 0.01    # AGC time constant in seconds
    track_update_rate: float = 10.0    # Track update rate in Hz
    
    # Performance characteristics
    range_resolution: float = 15.0     # Range resolution in meters
    velocity_resolution: float = 10.0  # Velocity resolution in Hz
    detection_threshold: float = 13.0  # Detection threshold in dB
    false_alarm_rate: float = 1e-6     # False alarm rate


@dataclass
class GateFeedback:
    """Gate position feedback from radar"""
    timestamp: float                   # Feedback timestamp
    range_gate_position: float         # Current range gate position (meters)
    velocity_gate_position: float      # Current velocity gate position (Hz)
    gate_status: GateStatus            # Gate tracking status
    lock_strength: float               # Lock strength (0-1)
    snr: float                        # Signal-to-noise ratio (dB)
    confidence: float                  # Position confidence (0-1)


class GatePullOff:
    """
    Advanced Gate Pull-Off System
    
    Implements sophisticated range and velocity gate pull-off techniques
    for radar jamming applications. Supports multiple pull-off profiles,
    adaptive rate calculation, and coordinated multi-target scenarios.
    """
    
    def __init__(self, 
                 sample_rate: float = 1e9,
                 processing_delay: float = 100e-9,
                 max_targets: int = 8):
        """
        Initialize gate pull-off system
        
        Args:
            sample_rate: System sampling rate in Hz
            processing_delay: Minimum processing delay in seconds
            max_targets: Maximum number of simultaneous targets
        """
        self.sample_rate = sample_rate
        self.processing_delay = processing_delay
        self.max_targets = max_targets
        self.c = 3e8  # Speed of light
        
        # System state
        self.is_active = True
        self.start_time = time.time()
        self.current_time = 0.0
        
        # Active pull-off operations
        self.active_range_pulls: Dict[str, Dict[str, Any]] = {}
        self.active_velocity_pulls: Dict[str, Dict[str, Any]] = {}
        self.active_combined_pulls: Dict[str, Dict[str, Any]] = {}
        
        # Feedback monitoring
        self.gate_feedback_history: List[GateFeedback] = []
        self.feedback_window_size = 100
        
        # Performance tracking
        self.success_rate = 0.0
        self.average_break_time = 0.0
        self.pull_off_attempts = 0
        self.successful_breaks = 0
        
        # Signal buffers
        self._range_delay_buffer = np.zeros(int(sample_rate * 0.01), dtype=complex)
        self._velocity_shift_buffer = np.zeros(int(sample_rate * 0.01), dtype=complex)
        
        # Adaptive control
        self._last_radar_params: Optional[RadarParams] = None
        self._optimal_rates: Dict[str, float] = {}
        
    def apply_range_gate_pull_off(self, 
                                 signal: np.ndarray,
                                 range_params: RangeParams,
                                 target_id: str = "default") -> np.ndarray:
        """
        Apply Range Gate Pull-Off (RGPO) technique
        
        Args:
            signal: Input complex signal samples
            range_params: Range pull-off parameters
            target_id: Unique identifier for this target
            
        Returns:
            Signal with range gate pull-off applied
        """
        if not self.is_active:
            return signal
            
        # Update current time
        self.current_time = len(signal) / self.sample_rate
        
        # Initialize or update pull-off state
        if target_id not in self.active_range_pulls:
            self.active_range_pulls[target_id] = {
                'start_time': self.current_time,
                'current_delay': range_params.initial_delay,
                'last_update': self.current_time,
                'params': range_params,
                'phase_reference': 0.0,
                'random_state': np.random.RandomState(hash(target_id) % 2**32)
            }
        
        pull_state = self.active_range_pulls[target_id]
        elapsed_time = self.current_time - pull_state['start_time']
        
        # Check if pull-off should continue
        if elapsed_time > range_params.duration:
            # Pull-off completed, maintain final position
            final_delay = pull_state['current_delay']
        else:
            # Calculate current delay based on profile
            final_delay = self._calculate_range_delay(
                elapsed_time, range_params, pull_state
            )
            pull_state['current_delay'] = final_delay
        
        # Apply range delay to signal
        delayed_signal = self._apply_range_delay(signal, final_delay)
        
        # Maintain phase coherence
        if range_params.profile != PullOffProfile.STEP:
            delayed_signal = self._maintain_range_phase_coherence(
                delayed_signal, pull_state, signal
            )
        
        return delayed_signal
    
    def apply_velocity_gate_pull_off(self, 
                                   signal: np.ndarray,
                                   velocity_params: VelocityParams,
                                   target_id: str = "default") -> np.ndarray:
        """
        Apply Velocity Gate Pull-Off (VGPO) technique
        
        Args:
            signal: Input complex signal samples
            velocity_params: Velocity pull-off parameters
            target_id: Unique identifier for this target
            
        Returns:
            Signal with velocity gate pull-off applied
        """
        if not self.is_active:
            return signal
            
        # Update current time
        self.current_time = len(signal) / self.sample_rate
        
        # Initialize or update pull-off state
        if target_id not in self.active_velocity_pulls:
            self.active_velocity_pulls[target_id] = {
                'start_time': self.current_time,
                'current_doppler': velocity_params.initial_doppler,
                'last_update': self.current_time,
                'params': velocity_params,
                'accumulated_phase': 0.0,
                'random_state': np.random.RandomState(hash(target_id + "_v") % 2**32)
            }
        
        pull_state = self.active_velocity_pulls[target_id]
        elapsed_time = self.current_time - pull_state['start_time']
        
        # Check if pull-off should continue
        if elapsed_time > velocity_params.duration:
            # Pull-off completed, maintain final velocity
            final_doppler = pull_state['current_doppler']
        else:
            # Calculate current Doppler shift based on profile
            final_doppler = self._calculate_doppler_shift(
                elapsed_time, velocity_params, pull_state
            )
            pull_state['current_doppler'] = final_doppler
        
        # Apply Doppler shift to signal
        shifted_signal = self._apply_doppler_shift(signal, final_doppler, pull_state)
        
        return shifted_signal
    
    def apply_combined_pull_off(self, 
                              signal: np.ndarray,
                              range_params: RangeParams,
                              velocity_params: VelocityParams,
                              target_id: str = "default") -> np.ndarray:
        """
        Apply combined range and velocity gate pull-off
        
        Args:
            signal: Input complex signal samples
            range_params: Range pull-off parameters
            velocity_params: Velocity pull-off parameters
            target_id: Unique identifier for this target
            
        Returns:
            Signal with combined pull-off applied
        """
        if not self.is_active:
            return signal
        
        # Apply range pull-off first
        range_signal = self.apply_range_gate_pull_off(
            signal, range_params, f"{target_id}_range"
        )
        
        # Then apply velocity pull-off
        combined_signal = self.apply_velocity_gate_pull_off(
            range_signal, velocity_params, f"{target_id}_velocity"
        )
        
        # Store combined state
        self.active_combined_pulls[target_id] = {
            'range_state': self.active_range_pulls.get(f"{target_id}_range"),
            'velocity_state': self.active_velocity_pulls.get(f"{target_id}_velocity"),
            'start_time': self.current_time
        }
        
        return combined_signal
    
    def calculate_optimal_pull_rate(self, 
                                  radar_params: RadarParams,
                                  pull_type: str = "range") -> float:
        """
        Calculate optimal pull rate based on radar parameters
        
        Args:
            radar_params: Radar system parameters
            pull_type: Type of pull-off ("range" or "velocity")
            
        Returns:
            Optimal pull rate in appropriate units (m/s or Hz/s)
        """
        self._last_radar_params = radar_params
        
        if pull_type == "range":
            # Calculate optimal range pull rate
            gate_width = radar_params.range_gate_width
            track_rate = radar_params.track_update_rate
            agc_time = radar_params.agc_time_constant
            
            # Rate should be fast enough to break lock but not so fast as to be obvious
            optimal_rate = gate_width / (4 * agc_time)  # Conservative estimate
            
            # Adjust based on track update rate
            if track_rate > 0:
                max_rate = gate_width * track_rate / 2
                optimal_rate = min(optimal_rate, max_rate)
            
            # Store for future use
            self._optimal_rates['range'] = optimal_rate
            return optimal_rate
            
        elif pull_type == "velocity":
            # Calculate optimal velocity pull rate
            gate_width = radar_params.velocity_gate_width
            track_rate = radar_params.track_update_rate
            prf = radar_params.prf
            agc_time = radar_params.agc_time_constant
            
            # Rate based on gate width and tracking characteristics
            optimal_rate = gate_width / (2 * agc_time) if agc_time > 0 else gate_width * 10
            
            # Consider PRF limitations
            max_doppler = prf / 4  # Unambiguous Doppler range
            max_rate = max_doppler * track_rate / 2 if track_rate > 0 else max_doppler
            optimal_rate = min(optimal_rate, max_rate)
            
            # Store for future use
            self._optimal_rates['velocity'] = optimal_rate
            return optimal_rate
            
        else:
            raise ValueError(f"Unknown pull type: {pull_type}")
    
    def maintain_skin_return(self, 
                           signal: np.ndarray,
                           transition_params: TransitionParams,
                           pull_strength: float = 0.5) -> np.ndarray:
        """
        Maintain skin return while transitioning to false return
        
        Args:
            signal: Original signal (skin return)
            transition_params: Transition parameters
            pull_strength: Strength of pull-off (0=skin only, 1=false only)
            
        Returns:
            Blended signal with smooth transition
        """
        if pull_strength <= 0:
            return signal
        elif pull_strength >= 1:
            # No skin return, use false target only
            return np.zeros_like(signal)
        
        # Create transition window
        window_samples = int(transition_params.transition_duration * self.sample_rate)
        window = self._create_transition_window(
            window_samples, transition_params.amplitude_taper
        )
        
        # Calculate blending weights
        skin_weight = 1.0 - pull_strength
        false_weight = pull_strength
        
        # Apply power ratio threshold
        if skin_weight < transition_params.power_ratio_threshold:
            skin_weight = 0.0
            false_weight = 1.0
        
        # Blend signals with smooth transition
        blended_signal = signal * skin_weight
        
        # Apply transition window at boundaries if needed
        if window_samples > 0 and window_samples < len(signal):
            # Smooth the transition region
            fade_start = int(len(signal) * (1 - transition_params.overlap_factor))
            fade_length = min(window_samples, len(signal) - fade_start)
            
            if fade_start + fade_length <= len(signal):
                fade_window = window[:fade_length]
                blended_signal[fade_start:fade_start + fade_length] *= fade_window
        
        return blended_signal
    
    def update_gate_feedback(self, feedback: GateFeedback) -> None:
        """
        Update gate position feedback from radar
        
        Args:
            feedback: Current gate feedback information
        """
        self.gate_feedback_history.append(feedback)
        
        # Maintain history window
        if len(self.gate_feedback_history) > self.feedback_window_size:
            self.gate_feedback_history.pop(0)
        
        # Update performance metrics
        self._update_performance_metrics(feedback)
        
        # Check for successful break
        if feedback.gate_status == GateStatus.BROKEN:
            self.successful_breaks += 1
            break_time = feedback.timestamp - self.start_time
            if self.successful_breaks > 1:
                self.average_break_time = (
                    (self.average_break_time * (self.successful_breaks - 1) + break_time) 
                    / self.successful_breaks
                )
            else:
                self.average_break_time = break_time
    
    def detect_pull_off_success(self, window_size: int = 10) -> bool:
        """
        Detect if pull-off has been successful
        
        Args:
            window_size: Number of recent feedback samples to analyze
            
        Returns:
            True if pull-off appears successful
        """
        if len(self.gate_feedback_history) < window_size:
            return False
        
        recent_feedback = self.gate_feedback_history[-window_size:]
        
        # Check for consistent gate break or degraded lock
        broken_count = sum(1 for fb in recent_feedback 
                          if fb.gate_status in [GateStatus.BROKEN, GateStatus.BREAKING])
        
        # Also consider lock strength degradation
        avg_lock_strength = np.mean([fb.lock_strength for fb in recent_feedback])
        
        # Success if majority are broken/breaking OR average lock strength is low
        return (broken_count >= window_size * 0.6) or (avg_lock_strength < 0.3)
    
    def get_recovery_strategy(self, 
                            current_feedback: GateFeedback) -> Dict[str, Any]:
        """
        Generate recovery strategy if lock is reacquired
        
        Args:
            current_feedback: Current gate feedback
            
        Returns:
            Dictionary containing recovery strategy parameters
        """
        strategy = {
            'action': 'maintain',
            'new_range_params': None,
            'new_velocity_params': None,
            'coordination_required': False
        }
        
        if current_feedback.gate_status == GateStatus.RECOVERING:
            # Radar is trying to reacquire - implement countermeasures
            if current_feedback.lock_strength > 0.7:
                # Strong recovery attempt - use aggressive counter-pull
                strategy['action'] = 'counter_pull'
                strategy['new_range_params'] = RangeParams(
                    pull_rate=self._optimal_rates.get('range', 200.0) * 2,
                    direction=PullOffDirection.BIDIRECTIONAL,
                    profile=PullOffProfile.RANDOM_WALK,
                    jitter_amplitude=50.0
                )
                strategy['new_velocity_params'] = VelocityParams(
                    pull_rate=self._optimal_rates.get('velocity', 2000.0) * 2,
                    direction=PullOffDirection.BIDIRECTIONAL,
                    profile=PullOffProfile.RANDOM_WALK,
                    jitter_amplitude=200.0
                )
            else:
                # Weak recovery - maintain current deception
                strategy['action'] = 'maintain'
        
        elif current_feedback.gate_status == GateStatus.LOCKED:
            # Lock reacquired - restart pull-off with different parameters
            strategy['action'] = 'restart'
            strategy['coordination_required'] = True
            
        return strategy
    
    def coordinate_multi_target_pull_off(self, 
                                       target_configs: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Coordinate pull-off across multiple targets
        
        Args:
            target_configs: Dictionary mapping target IDs to their configurations
                          Each config should contain 'signal', 'range_params', 'velocity_params'
            
        Returns:
            Dictionary mapping target IDs to their processed signals
        """
        processed_signals = {}
        
        # Sort targets by priority (if specified)
        sorted_targets = sorted(
            target_configs.items(),
            key=lambda x: x[1].get('priority', 0),
            reverse=True
        )
        
        # Calculate timing coordination
        base_time = self.current_time
        time_offsets = self._calculate_coordination_timing(sorted_targets)
        
        for i, (target_id, config) in enumerate(sorted_targets):
            signal = config['signal']
            range_params = config.get('range_params')
            velocity_params = config.get('velocity_params')
            
            # Apply time offset for coordination
            if i < len(time_offsets):
                time_offset = time_offsets[i]
                # Adjust parameters for coordinated timing
                if range_params:
                    range_params.initial_delay += time_offset * range_params.pull_rate
                if velocity_params:
                    velocity_params.initial_doppler += time_offset * velocity_params.pull_rate
            
            # Apply appropriate pull-off technique
            if range_params and velocity_params:
                processed_signal = self.apply_combined_pull_off(
                    signal, range_params, velocity_params, target_id
                )
            elif range_params:
                processed_signal = self.apply_range_gate_pull_off(
                    signal, range_params, target_id
                )
            elif velocity_params:
                processed_signal = self.apply_velocity_gate_pull_off(
                    signal, velocity_params, target_id
                )
            else:
                processed_signal = signal
            
            processed_signals[target_id] = processed_signal
        
        return processed_signals
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'is_active': self.is_active,
            'active_range_pulls': len(self.active_range_pulls),
            'active_velocity_pulls': len(self.active_velocity_pulls),
            'active_combined_pulls': len(self.active_combined_pulls),
            'success_rate': self.success_rate,
            'average_break_time': self.average_break_time,
            'pull_off_attempts': self.pull_off_attempts,
            'successful_breaks': self.successful_breaks,
            'feedback_samples': len(self.gate_feedback_history),
            'optimal_range_rate': self._optimal_rates.get('range', 0.0),
            'optimal_velocity_rate': self._optimal_rates.get('velocity', 0.0),
            'current_time': self.current_time
        }
    
    # Private helper methods
    
    def _calculate_range_delay(self, 
                             elapsed_time: float,
                             params: RangeParams,
                             state: Dict[str, Any]) -> float:
        """Calculate current range delay based on profile"""
        if params.profile == PullOffProfile.LINEAR:
            # Linear pull at constant rate
            rate = params.pull_rate
            if params.direction == PullOffDirection.PULL_BACK:
                rate = -rate
            delay = params.initial_delay + rate * elapsed_time
            
        elif params.profile == PullOffProfile.EXPONENTIAL:
            # Exponential acceleration
            rate = params.pull_rate
            accel = params.acceleration
            if params.direction == PullOffDirection.PULL_BACK:
                rate = -rate
                accel = -accel
            delay = (params.initial_delay + 
                    rate * elapsed_time + 
                    0.5 * accel * elapsed_time**2)
            
        elif params.profile == PullOffProfile.S_CURVE:
            # S-curve using sigmoid function
            progress = elapsed_time / params.duration
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            total_pull = params.pull_rate * params.duration
            if params.direction == PullOffDirection.PULL_BACK:
                total_pull = -total_pull
            delay = params.initial_delay + total_pull * sigmoid
            
        elif params.profile == PullOffProfile.STEP:
            # Discrete steps
            step_number = int(elapsed_time / params.step_interval)
            step_delay = step_number * params.step_size
            if params.direction == PullOffDirection.PULL_BACK:
                step_delay = -step_delay
            delay = params.initial_delay + step_delay
            
        elif params.profile == PullOffProfile.RANDOM_WALK:
            # Random walk with trend
            if 'random_steps' not in state:
                state['random_steps'] = []
            
            # Add new random step
            jitter = (state['random_state'].normal(0, params.jitter_amplitude) 
                     if params.jitter_amplitude > 0 else 0)
            base_rate = params.pull_rate
            if params.direction == PullOffDirection.PULL_BACK:
                base_rate = -base_rate
            
            random_step = base_rate * (elapsed_time / params.duration) + jitter
            state['random_steps'].append(random_step)
            
            delay = params.initial_delay + sum(state['random_steps']) / len(state['random_steps'])
            
        else:
            # Default to linear
            delay = params.initial_delay + params.pull_rate * elapsed_time
        
        # Apply bounds
        delay = np.clip(delay, params.min_delay, params.max_delay)
        
        return delay
    
    def _calculate_doppler_shift(self, 
                               elapsed_time: float,
                               params: VelocityParams,
                               state: Dict[str, Any]) -> float:
        """Calculate current Doppler shift based on profile"""
        if params.profile == PullOffProfile.LINEAR:
            # Linear pull at constant rate
            rate = params.pull_rate
            if params.direction == PullOffDirection.PULL_DOWN:
                rate = -rate
            doppler = params.initial_doppler + rate * elapsed_time
            
        elif params.profile == PullOffProfile.EXPONENTIAL:
            # Exponential acceleration
            rate = params.pull_rate
            accel = params.acceleration
            if params.direction == PullOffDirection.PULL_DOWN:
                rate = -rate
                accel = -accel
            doppler = (params.initial_doppler + 
                      rate * elapsed_time + 
                      0.5 * accel * elapsed_time**2)
            
        elif params.profile == PullOffProfile.S_CURVE:
            # S-curve using sigmoid function
            progress = elapsed_time / params.duration
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            total_pull = params.pull_rate * params.duration
            if params.direction == PullOffDirection.PULL_DOWN:
                total_pull = -total_pull
            doppler = params.initial_doppler + total_pull * sigmoid
            
        elif params.profile == PullOffProfile.STEP:
            # Discrete steps
            step_number = int(elapsed_time / params.step_interval)
            step_shift = step_number * params.step_size
            if params.direction == PullOffDirection.PULL_DOWN:
                step_shift = -step_shift
            doppler = params.initial_doppler + step_shift
            
        elif params.profile == PullOffProfile.RANDOM_WALK:
            # Random walk with trend
            if 'random_steps' not in state:
                state['random_steps'] = []
            
            # Add new random step
            jitter = (state['random_state'].normal(0, params.jitter_amplitude) 
                     if params.jitter_amplitude > 0 else 0)
            base_rate = params.pull_rate
            if params.direction == PullOffDirection.PULL_DOWN:
                base_rate = -base_rate
            
            random_step = base_rate * (elapsed_time / params.duration) + jitter
            state['random_steps'].append(random_step)
            
            doppler = params.initial_doppler + sum(state['random_steps']) / len(state['random_steps'])
            
        else:
            # Default to linear
            doppler = params.initial_doppler + params.pull_rate * elapsed_time
        
        # Apply bounds
        doppler = np.clip(doppler, params.min_doppler, params.max_doppler)
        
        return doppler
    
    def _apply_range_delay(self, signal: np.ndarray, delay_meters: float) -> np.ndarray:
        """Apply range delay to signal"""
        # Convert range delay to time delay (round trip)
        time_delay = 2 * delay_meters / self.c
        delay_samples = int(time_delay * self.sample_rate)
        
        if delay_samples == 0:
            return signal
        elif delay_samples > 0:
            # Positive delay - shift signal later in time
            delayed_signal = np.zeros_like(signal)
            if delay_samples < len(signal):
                delayed_signal[delay_samples:] = signal[:-delay_samples]
        else:
            # Negative delay - shift signal earlier in time
            delayed_signal = np.zeros_like(signal)
            abs_delay = abs(delay_samples)
            if abs_delay < len(signal):
                delayed_signal[:-abs_delay] = signal[abs_delay:]
        
        return delayed_signal
    
    def _apply_doppler_shift(self, 
                           signal: np.ndarray, 
                           doppler_hz: float,
                           state: Dict[str, Any]) -> np.ndarray:
        """Apply Doppler shift to signal"""
        if doppler_hz == 0:
            return signal
        
        # Generate time vector
        time_vector = np.arange(len(signal)) / self.sample_rate
        
        # Calculate phase progression
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1/self.sample_rate
        phase_increment = 2 * np.pi * doppler_hz * dt
        
        # Maintain phase continuity
        if 'accumulated_phase' in state:
            phase_offset = state['accumulated_phase']
        else:
            phase_offset = 0.0
        
        # Generate Doppler shift phasor
        phase_vector = phase_offset + 2 * np.pi * doppler_hz * time_vector
        doppler_phasor = np.exp(1j * phase_vector)
        
        # Update accumulated phase for next call
        state['accumulated_phase'] = phase_vector[-1] % (2 * np.pi)
        
        # Apply Doppler shift
        shifted_signal = signal * doppler_phasor
        
        return shifted_signal
    
    def _maintain_range_phase_coherence(self, 
                                      signal: np.ndarray,
                                      state: Dict[str, Any],
                                      original_signal: np.ndarray) -> np.ndarray:
        """Maintain phase coherence during range pull-off"""
        # Calculate phase correction based on delay change
        current_delay = state.get('current_delay', 0.0)
        last_delay = state.get('last_delay', current_delay)
        
        delay_change = current_delay - last_delay
        time_delay_change = 2 * delay_change / self.c
        
        # Initialize phase correction
        phase_correction = 0.0
        
        # Phase correction for delay change
        if self._last_radar_params:
            carrier_freq = self._last_radar_params.carrier_frequency
            phase_correction = 2 * np.pi * carrier_freq * time_delay_change
            
            # Apply gradual phase correction to maintain coherence
            if abs(phase_correction) > 0.1:  # Only correct significant changes
                correction_phasor = np.exp(-1j * phase_correction)
                signal = signal * correction_phasor
        
        # Update state
        state['last_delay'] = current_delay
        state['phase_reference'] = state.get('phase_reference', 0.0) + phase_correction
        
        return signal
    
    def _create_transition_window(self, window_length: int, window_type: str) -> np.ndarray:
        """Create transition window for smooth amplitude transitions"""
        if window_length <= 0:
            return np.array([1.0])
        
        if window_type == "hanning":
            return np.hanning(window_length)
        elif window_type == "hamming":
            return np.hamming(window_length)
        elif window_type == "blackman":
            return np.blackman(window_length)
        elif window_type == "kaiser":
            return np.kaiser(window_length, beta=8.0)
        else:
            # Linear taper
            return np.linspace(1.0, 0.0, window_length)
    
    def _update_performance_metrics(self, feedback: GateFeedback) -> None:
        """Update performance metrics based on feedback"""
        self.pull_off_attempts += 1
        
        # Calculate success rate
        if self.pull_off_attempts > 0:
            self.success_rate = self.successful_breaks / self.pull_off_attempts
    
    def _calculate_coordination_timing(self, 
                                     target_list: List[Tuple[str, Dict[str, Any]]]) -> List[float]:
        """Calculate timing offsets for coordinated multi-target pull-off"""
        num_targets = len(target_list)
        if num_targets <= 1:
            return [0.0]
        
        # Stagger pull-offs to avoid simultaneous breaks that might alert radar
        base_interval = 0.1  # 100ms between initiations
        time_offsets = [i * base_interval for i in range(num_targets)]
        
        # Add some randomization to make it less predictable
        jitter = np.random.normal(0, base_interval * 0.2, num_targets)
        time_offsets = [max(0, offset + j) for offset, j in zip(time_offsets, jitter)]
        
        return time_offsets


def create_adaptive_range_params(radar_params: RadarParams, 
                               aggressiveness: float = 0.5) -> RangeParams:
    """
    Create adaptive range parameters based on radar characteristics
    
    Args:
        radar_params: Radar system parameters
        aggressiveness: Pull-off aggressiveness (0=conservative, 1=aggressive)
        
    Returns:
        Optimized RangeParams
    """
    # Calculate optimal pull rate
    gate_width = radar_params.range_gate_width
    track_rate = radar_params.track_update_rate
    agc_time = radar_params.agc_time_constant
    
    # Base pull rate on gate characteristics
    base_rate = gate_width / (4 * agc_time) if agc_time > 0 else gate_width * 10
    
    # Adjust for aggressiveness
    pull_rate = base_rate * (0.5 + aggressiveness)
    
    # Choose profile based on radar characteristics
    if radar_params.prf > 5000:  # High PRF - use smooth profiles
        profile = PullOffProfile.S_CURVE
    elif aggressiveness > 0.7:  # High aggressiveness - use random walk
        profile = PullOffProfile.RANDOM_WALK
    else:
        profile = PullOffProfile.LINEAR
    
    return RangeParams(
        pull_rate=pull_rate,
        profile=profile,
        max_delay=gate_width * 3,  # Pull to 3x gate width
        duration=1.0 / track_rate if track_rate > 0 else 1.0,
        jitter_amplitude=gate_width * 0.1 if profile == PullOffProfile.RANDOM_WALK else 0.0
    )


def create_adaptive_velocity_params(radar_params: RadarParams, 
                                  aggressiveness: float = 0.5) -> VelocityParams:
    """
    Create adaptive velocity parameters based on radar characteristics
    
    Args:
        radar_params: Radar system parameters
        aggressiveness: Pull-off aggressiveness (0=conservative, 1=aggressive)
        
    Returns:
        Optimized VelocityParams
    """
    # Calculate optimal pull rate
    gate_width = radar_params.velocity_gate_width
    track_rate = radar_params.track_update_rate
    prf = radar_params.prf
    
    # Base pull rate on gate characteristics
    base_rate = gate_width / (2 * radar_params.agc_time_constant) if radar_params.agc_time_constant > 0 else gate_width * 20
    
    # Adjust for aggressiveness
    pull_rate = base_rate * (0.5 + aggressiveness)
    
    # Consider PRF limitations
    max_unambiguous_doppler = prf / 4
    pull_rate = min(pull_rate, max_unambiguous_doppler / 2)
    
    # Choose profile based on characteristics
    if aggressiveness > 0.8:
        profile = PullOffProfile.RANDOM_WALK
    elif radar_params.prf > 3000:
        profile = PullOffProfile.S_CURVE
    else:
        profile = PullOffProfile.LINEAR
    
    return VelocityParams(
        pull_rate=pull_rate,
        profile=profile,
        max_doppler=max_unambiguous_doppler * 0.8,
        min_doppler=-max_unambiguous_doppler * 0.8,
        duration=1.0 / track_rate if track_rate > 0 else 1.0,
        jitter_amplitude=gate_width * 0.15 if profile == PullOffProfile.RANDOM_WALK else 0.0
    )


def calculate_pull_off_effectiveness(feedback_history: List[GateFeedback],
                                   pull_start_time: float) -> Dict[str, float]:
    """
    Calculate pull-off effectiveness metrics
    
    Args:
        feedback_history: Historical gate feedback data
        pull_start_time: When pull-off was initiated
        
    Returns:
        Dictionary of effectiveness metrics
    """
    if not feedback_history:
        return {'effectiveness': 0.0, 'break_time': float('inf'), 'lock_degradation': 0.0}
    
    # Find feedback after pull-off start
    relevant_feedback = [fb for fb in feedback_history if fb.timestamp >= pull_start_time]
    
    if not relevant_feedback:
        return {'effectiveness': 0.0, 'break_time': float('inf'), 'lock_degradation': 0.0}
    
    # Calculate break time
    break_time = float('inf')
    for fb in relevant_feedback:
        if fb.gate_status in [GateStatus.BROKEN, GateStatus.BREAKING]:
            break_time = fb.timestamp - pull_start_time
            break
    
    # Calculate lock strength degradation
    initial_lock = relevant_feedback[0].lock_strength
    min_lock = min(fb.lock_strength for fb in relevant_feedback)
    lock_degradation = max(0, initial_lock - min_lock)
    
    # Calculate overall effectiveness
    time_factor = 1.0 if break_time < 1.0 else max(0, 2.0 - break_time)
    lock_factor = lock_degradation
    effectiveness = (time_factor + lock_factor) / 2.0
    
    return {
        'effectiveness': effectiveness,
        'break_time': break_time,
        'lock_degradation': lock_degradation,
        'final_status': relevant_feedback[-1].gate_status.value if relevant_feedback else 'unknown'
    }