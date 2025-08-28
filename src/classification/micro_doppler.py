#!/usr/bin/env python3
"""
Micro-Doppler signature simulation for different target types
Simulates rotating parts, flapping wings, and other micro-motions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MicroDopplerType(Enum):
    """Types of micro-Doppler signatures"""
    ROTOR_BLADE = "rotor_blade"      # Helicopter main rotor
    PROPELLER = "propeller"          # Aircraft propeller
    JET_ENGINE = "jet_engine"        # Jet turbine
    DRONE_ROTORS = "drone_rotors"    # Quadcopter rotors
    BIRD_WINGS = "bird_wings"        # Flapping wings
    HUMAN_GAIT = "human_gait"        # Walking human
    VEHICLE_WHEELS = "vehicle_wheels" # Rotating wheels
    NONE = "none"                    # No micro-Doppler


@dataclass
class MicroDopplerParams:
    """Parameters for micro-Doppler simulation"""
    type: MicroDopplerType
    rotation_rate: float = 0.0        # Hz (rotations/flaps per second)
    num_blades: int = 2               # Number of blades/wings
    blade_length: float = 5.0         # meters
    modulation_strength: float = 1.0  # Relative strength
    harmonics: int = 3                # Number of harmonics to include
    
    # Additional parameters for specific types
    flapping_angle: float = 30.0      # degrees (for birds)
    blade_pitch: float = 10.0         # degrees (for rotors)
    turbine_stages: int = 1           # Number of turbine stages


class MicroDopplerSimulator:
    """Simulate micro-Doppler effects for various target types"""
    
    def __init__(self, radar_frequency: float = 10e9, sampling_rate: float = 1000):
        """
        Initialize micro-Doppler simulator
        
        Args:
            radar_frequency: Radar carrier frequency (Hz)
            sampling_rate: Sampling rate for micro-Doppler (Hz)
        """
        self.radar_frequency = radar_frequency
        self.wavelength = 3e8 / radar_frequency
        self.sampling_rate = sampling_rate
        
        # Pre-defined parameters for common targets
        self.target_library = self._initialize_target_library()
    
    def _initialize_target_library(self) -> Dict[str, MicroDopplerParams]:
        """Initialize library of common target micro-Doppler parameters"""
        return {
            'helicopter': MicroDopplerParams(
                type=MicroDopplerType.ROTOR_BLADE,
                rotation_rate=4.5,  # ~270 RPM
                num_blades=4,
                blade_length=8.0,
                modulation_strength=2.0
            ),
            'propeller_aircraft': MicroDopplerParams(
                type=MicroDopplerType.PROPELLER,
                rotation_rate=40,  # ~2400 RPM
                num_blades=3,
                blade_length=1.5,
                modulation_strength=1.5
            ),
            'jet_fighter': MicroDopplerParams(
                type=MicroDopplerType.JET_ENGINE,
                rotation_rate=200,  # High-speed turbine
                num_blades=20,  # Turbine blades
                blade_length=0.3,
                modulation_strength=0.5,
                turbine_stages=2
            ),
            'quadcopter': MicroDopplerParams(
                type=MicroDopplerType.DRONE_ROTORS,
                rotation_rate=100,  # ~6000 RPM for small props
                num_blades=2,
                blade_length=0.15,
                modulation_strength=0.8,
                harmonics=4  # 4 rotors create harmonics
            ),
            'bird': MicroDopplerParams(
                type=MicroDopplerType.BIRD_WINGS,
                rotation_rate=5,  # 5 Hz wing beat
                num_blades=2,  # Two wings
                blade_length=0.3,  # Wing span
                flapping_angle=45,
                modulation_strength=1.0
            ),
            'walking_human': MicroDopplerParams(
                type=MicroDopplerType.HUMAN_GAIT,
                rotation_rate=2,  # ~2 steps/second
                num_blades=4,  # Arms and legs
                blade_length=0.8,  # Limb length
                modulation_strength=0.3
            ),
            'ground_vehicle': MicroDopplerParams(
                type=MicroDopplerType.VEHICLE_WHEELS,
                rotation_rate=10,  # Wheel rotation at ~30 mph
                num_blades=4,  # 4 wheels
                blade_length=0.35,  # Wheel radius
                modulation_strength=0.2
            )
        }
    
    def generate_micro_doppler(self, 
                              target_type: str,
                              target_velocity: float,
                              duration: float = 1.0,
                              aspect_angle: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate micro-Doppler signature for a target
        
        Args:
            target_type: Type of target ('helicopter', 'quadcopter', etc.)
            target_velocity: Bulk velocity of target (m/s)
            duration: Duration of signature (seconds)
            aspect_angle: Aspect angle to target (degrees)
            
        Returns:
            time_array: Time samples
            doppler_signature: Doppler frequency over time
        """
        if target_type not in self.target_library:
            # Return just bulk Doppler
            time_array = np.linspace(0, duration, int(duration * self.sampling_rate))
            bulk_doppler = 2 * target_velocity / self.wavelength
            return time_array, np.full_like(time_array, bulk_doppler)
        
        params = self.target_library[target_type]
        time_array = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Bulk Doppler shift
        bulk_doppler = 2 * target_velocity / self.wavelength
        
        # Generate micro-Doppler based on type
        if params.type == MicroDopplerType.ROTOR_BLADE:
            micro_doppler = self._generate_rotor_doppler(params, time_array, aspect_angle)
        elif params.type == MicroDopplerType.DRONE_ROTORS:
            micro_doppler = self._generate_multirotor_doppler(params, time_array, aspect_angle)
        elif params.type == MicroDopplerType.BIRD_WINGS:
            micro_doppler = self._generate_flapping_doppler(params, time_array, aspect_angle)
        elif params.type == MicroDopplerType.JET_ENGINE:
            micro_doppler = self._generate_turbine_doppler(params, time_array, aspect_angle)
        elif params.type == MicroDopplerType.HUMAN_GAIT:
            micro_doppler = self._generate_gait_doppler(params, time_array, aspect_angle)
        else:
            micro_doppler = self._generate_generic_rotation_doppler(params, time_array, aspect_angle)
        
        # Combine bulk and micro-Doppler
        total_doppler = bulk_doppler + micro_doppler
        
        return time_array, total_doppler
    
    def _generate_rotor_doppler(self, params: MicroDopplerParams, 
                               time_array: np.ndarray,
                               aspect_angle: float) -> np.ndarray:
        """Generate helicopter rotor micro-Doppler"""
        
        micro_doppler = np.zeros_like(time_array)
        aspect_rad = np.radians(aspect_angle)
        
        for blade in range(params.num_blades):
            blade_phase = 2 * np.pi * blade / params.num_blades
            
            # Blade tip velocity
            tip_velocity = 2 * np.pi * params.rotation_rate * params.blade_length
            
            # Time-varying Doppler from rotating blade
            blade_angle = 2 * np.pi * params.rotation_rate * time_array + blade_phase
            
            # Doppler shift varies sinusoidally with blade rotation
            blade_doppler = (2 * tip_velocity / self.wavelength) * \
                          np.sin(blade_angle) * np.cos(aspect_rad)
            
            # Add blade flash (strong return when perpendicular)
            flash_modulation = np.exp(-((blade_angle % (2*np.pi) - np.pi/2)**2) / 0.1)
            
            micro_doppler += params.modulation_strength * blade_doppler * (1 + flash_modulation)
        
        # Add harmonics
        for harmonic in range(2, params.harmonics + 1):
            harmonic_doppler = (0.3 / harmonic) * np.sin(2 * np.pi * harmonic * 
                                                         params.rotation_rate * params.num_blades * time_array)
            micro_doppler += harmonic_doppler * (2 * params.blade_length / self.wavelength)
        
        return micro_doppler
    
    def _generate_multirotor_doppler(self, params: MicroDopplerParams, 
                                    time_array: np.ndarray,
                                    aspect_angle: float) -> np.ndarray:
        """Generate quadcopter/drone micro-Doppler (multiple rotors)"""
        
        micro_doppler = np.zeros_like(time_array)
        num_rotors = 4  # Typical quadcopter
        
        for rotor in range(num_rotors):
            # Each rotor spins at slightly different rate (realistic)
            rotor_rate = params.rotation_rate * (1 + 0.02 * np.sin(rotor))
            rotor_phase = 2 * np.pi * rotor / num_rotors
            
            # Counter-rotating pairs
            direction = 1 if rotor % 2 == 0 else -1
            
            for blade in range(params.num_blades):
                blade_phase = 2 * np.pi * blade / params.num_blades
                
                # Blade tip velocity
                tip_velocity = direction * 2 * np.pi * rotor_rate * params.blade_length
                
                # Time-varying Doppler
                blade_angle = 2 * np.pi * rotor_rate * time_array + blade_phase + rotor_phase
                blade_doppler = (2 * tip_velocity / self.wavelength) * np.sin(blade_angle)
                
                micro_doppler += params.modulation_strength * blade_doppler / num_rotors
        
        # Add high-frequency hash from multiple rotors
        hash_freq = params.rotation_rate * params.num_blades * num_rotors
        micro_doppler += 0.1 * params.modulation_strength * \
                        np.sin(2 * np.pi * hash_freq * time_array) * \
                        (2 * params.blade_length / self.wavelength)
        
        return micro_doppler
    
    def _generate_flapping_doppler(self, params: MicroDopplerParams, 
                                  time_array: np.ndarray,
                                  aspect_angle: float) -> np.ndarray:
        """Generate bird wing flapping micro-Doppler"""
        
        # Flapping is more complex than rotation
        flap_angle_rad = np.radians(params.flapping_angle)
        
        # Wing beat pattern (not perfectly sinusoidal)
        wing_phase = 2 * np.pi * params.rotation_rate * time_array
        
        # Downstroke is faster than upstroke
        wing_motion = np.where(np.sin(wing_phase) > 0,
                              np.sin(wing_phase),  # Downstroke
                              0.7 * np.sin(wing_phase))  # Slower upstroke
        
        # Wing tip velocity
        wing_velocity = params.rotation_rate * params.blade_length * flap_angle_rad
        
        # Doppler modulation
        micro_doppler = (2 * wing_velocity / self.wavelength) * wing_motion * \
                       params.modulation_strength
        
        # Add body oscillation (bird body moves up/down with wing beats)
        body_motion = 0.2 * np.sin(2 * wing_phase)  # Double frequency
        micro_doppler += (2 * 0.5 / self.wavelength) * body_motion
        
        # Add some randomness (biological variation)
        micro_doppler += 0.05 * np.random.randn(len(time_array)) * \
                        (2 * wing_velocity / self.wavelength)
        
        return micro_doppler
    
    def _generate_turbine_doppler(self, params: MicroDopplerParams, 
                                 time_array: np.ndarray,
                                 aspect_angle: float) -> np.ndarray:
        """Generate jet engine turbine micro-Doppler"""
        
        micro_doppler = np.zeros_like(time_array)
        
        # Multiple turbine stages at different speeds
        for stage in range(params.turbine_stages):
            stage_rate = params.rotation_rate * (1 - 0.3 * stage)  # Each stage slower
            
            # Many blades create a nearly continuous modulation
            for blade in range(params.num_blades):
                blade_phase = 2 * np.pi * blade / params.num_blades
                
                # Small blade length for turbine
                blade_velocity = 2 * np.pi * stage_rate * params.blade_length
                
                # High-frequency modulation
                blade_angle = 2 * np.pi * stage_rate * time_array + blade_phase
                blade_doppler = (2 * blade_velocity / self.wavelength) * \
                              np.sin(blade_angle) * np.cos(np.radians(aspect_angle))
                
                micro_doppler += params.modulation_strength * blade_doppler / \
                               (params.num_blades * params.turbine_stages)
        
        # Add high-frequency carrier from turbine
        carrier_freq = params.rotation_rate * params.num_blades
        micro_doppler += 0.05 * np.sin(2 * np.pi * carrier_freq * time_array) * \
                        (2 * params.blade_length / self.wavelength)
        
        return micro_doppler
    
    def _generate_gait_doppler(self, params: MicroDopplerParams, 
                              time_array: np.ndarray,
                              aspect_angle: float) -> np.ndarray:
        """Generate human walking gait micro-Doppler"""
        
        # Gait cycle
        gait_phase = 2 * np.pi * params.rotation_rate * time_array
        
        # Torso motion (slight bobbing)
        torso_motion = 0.1 * np.sin(2 * gait_phase)  # Double frequency
        
        # Leg motion (alternating)
        leg1_motion = np.maximum(0, np.sin(gait_phase))  # Forward swing
        leg2_motion = np.maximum(0, np.sin(gait_phase + np.pi))  # Opposite phase
        
        # Arm swing (opposite to legs)
        arm1_motion = 0.5 * np.sin(gait_phase + np.pi)
        arm2_motion = 0.5 * np.sin(gait_phase)
        
        # Combine limb motions
        limb_velocity = params.rotation_rate * params.blade_length  # Limb swing velocity
        
        micro_doppler = (2 / self.wavelength) * params.modulation_strength * (
            torso_motion * 0.5 +
            (leg1_motion + leg2_motion) * limb_velocity * 0.4 +
            (arm1_motion + arm2_motion) * limb_velocity * 0.2
        )
        
        return micro_doppler
    
    def _generate_generic_rotation_doppler(self, params: MicroDopplerParams, 
                                          time_array: np.ndarray,
                                          aspect_angle: float) -> np.ndarray:
        """Generate generic rotating part micro-Doppler"""
        
        micro_doppler = np.zeros_like(time_array)
        
        for component in range(params.num_blades):
            phase = 2 * np.pi * component / params.num_blades
            rotation_angle = 2 * np.pi * params.rotation_rate * time_array + phase
            
            # Simple sinusoidal modulation
            component_velocity = 2 * np.pi * params.rotation_rate * params.blade_length
            component_doppler = (2 * component_velocity / self.wavelength) * \
                              np.sin(rotation_angle) * np.cos(np.radians(aspect_angle))
            
            micro_doppler += params.modulation_strength * component_doppler
        
        return micro_doppler
    
    def create_spectrogram(self, time_array: np.ndarray, 
                          doppler_signature: np.ndarray,
                          window_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create micro-Doppler spectrogram from time-domain signature
        
        Args:
            time_array: Time samples
            doppler_signature: Doppler frequency over time
            window_size: STFT window size
            
        Returns:
            time_bins: Time axis for spectrogram
            frequency_bins: Frequency axis for spectrogram
            spectrogram: 2D spectrogram array
        """
        from scipy import signal
        
        # Convert Doppler frequency to complex signal
        complex_signal = np.exp(1j * 2 * np.pi * np.cumsum(doppler_signature) / self.sampling_rate)
        
        # Compute STFT
        f, t, Sxx = signal.spectrogram(complex_signal, 
                                       fs=self.sampling_rate,
                                       window='hann',
                                       nperseg=window_size,
                                       noverlap=window_size//2)
        
        # Convert to dB scale
        spectrogram_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
        
        return t, f, spectrogram_db