"""
RF propagation channel modeling for radar simulation with realistic physics-based effects.

This module provides:
- Free space path loss with R^4 radar equation
- Target RCS with Swerling fluctuation models
- Doppler shifts from target motion
- Atmospheric attenuation effects
- Multipath propagation
- Realistic clutter returns

CRITICAL: This module uses only physics-based calculations and does NOT access
true target positions directly for realistic simulation.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy.special import erfc
import warnings

# Import base classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar import RadarParameters
from target import Target, TargetType
from environment import Environment, AtmosphericConditions, WeatherType


class SwerlingModel(Enum):
    """Swerling fluctuation models"""
    SWERLING_0 = "constant"  # Non-fluctuating
    SWERLING_1 = "swerling1"  # Slow fluctuation, exponential
    SWERLING_2 = "swerling2"  # Fast fluctuation, exponential  
    SWERLING_3 = "swerling3"  # Slow fluctuation, chi-squared 4 DOF
    SWERLING_4 = "swerling4"  # Fast fluctuation, chi-squared 4 DOF


class PropagationMode(Enum):
    """Propagation modes"""
    FREE_SPACE = "free_space"
    TWO_RAY = "two_ray"
    MULTIPATH = "multipath"
    DUCTING = "ducting"


@dataclass
class ClutterParameters:
    """Clutter modeling parameters"""
    clutter_type: str = "land"  # 'land', 'sea', 'urban', 'weather'
    reflectivity_db: float = -20.0  # Average sigma-0 in dB
    decorrelation_time: float = 0.1  # Clutter decorrelation time (seconds)
    spectral_width: float = 2.0  # Doppler spectral width (m/s)
    range_extent: float = 100.0  # Clutter range extent (meters)
    azimuth_extent: float = 0.02  # Clutter azimuth extent (radians)
    temporal_correlation: float = 0.8  # Temporal correlation coefficient


@dataclass
class MultipathParameters:
    """Multipath propagation parameters"""
    num_paths: int = 5  # Number of multipath components
    path_delays: Optional[np.ndarray] = None  # Path delays (seconds)
    path_gains: Optional[np.ndarray] = None  # Path gains (linear)
    path_phases: Optional[np.ndarray] = None  # Path phases (radians)
    surface_roughness: float = 0.1  # Surface roughness (meters)
    terrain_profile: Optional[np.ndarray] = None  # Terrain height profile


class PropagationChannel:
    """Comprehensive RF propagation channel model for radar simulation"""
    
    def __init__(self,
                 radar_params: RadarParameters,
                 environment: Optional[Environment] = None,
                 propagation_mode: PropagationMode = PropagationMode.FREE_SPACE,
                 enable_fluctuations: bool = True):
        """
        Initialize propagation channel
        
        Args:
            radar_params: Radar system parameters
            environment: Environmental conditions
            propagation_mode: Propagation modeling approach
            enable_fluctuations: Enable target fluctuation models
        """
        self.radar_params = radar_params
        self.environment = environment or Environment()
        self.propagation_mode = propagation_mode
        self.enable_fluctuations = enable_fluctuations
        
        # Physical constants
        self.c = const.c  # Speed of light
        self.k_boltzmann = const.k  # Boltzmann constant
        
        # Internal state for fluctuations
        self._fluctuation_state = {}
        self._clutter_memory = {}
        
    def radar_equation(self,
                      range_m: float,
                      target_rcs: float,
                      target_velocity: Optional[float] = None,
                      swerling_model: SwerlingModel = SwerlingModel.SWERLING_1,
                      include_losses: bool = True) -> float:
        """
        Calculate received power using the radar equation with R^4 dependence
        
        Args:
            range_m: Range to target in meters
            target_rcs: Target radar cross section in m^2
            target_velocity: Target radial velocity in m/s (for Doppler effects)
            swerling_model: Swerling fluctuation model
            include_losses: Include propagation and system losses
            
        Returns:
            Received power in Watts
        """
        # Basic radar equation: Pr = (Pt * Gt * Gr * λ² * σ) / ((4π)³ * R⁴ * L)
        
        # Transmit power
        pt = self.radar_params.power
        
        # Antenna gains (linear)
        gt = 10**(self.radar_params.antenna_gain / 10)
        gr = gt  # Monostatic radar assumption
        
        # Wavelength
        wavelength = self.c / self.radar_params.frequency
        
        # Apply fluctuation model to RCS
        fluctuating_rcs = self._apply_swerling_fluctuation(target_rcs, swerling_model)
        
        # Calculate losses
        if include_losses:
            # System losses
            system_losses = 10**(self.radar_params.losses / 10)
            
            # Propagation losses
            prop_losses = self._calculate_propagation_losses(range_m, target_velocity)
            
            total_losses = system_losses * prop_losses
        else:
            total_losses = 1.0
        
        # Radar equation calculation
        numerator = pt * gt * gr * wavelength**2 * fluctuating_rcs
        denominator = (4 * np.pi)**3 * range_m**4 * total_losses
        
        received_power = numerator / denominator
        
        return received_power
    
    def calculate_doppler_shift(self,
                               target_velocity: float,
                               target_position: Optional[np.ndarray] = None,
                               radar_position: Optional[np.ndarray] = None,
                               relativistic_correction: bool = False) -> float:
        """
        Calculate Doppler shift from target motion
        
        Args:
            target_velocity: Radial velocity in m/s (positive = approaching)
            target_position: Target position vector (for geometry calculations)
            radar_position: Radar position vector
            relativistic_correction: Apply relativistic correction for high speeds
            
        Returns:
            Doppler shift in Hz
        """
        # Basic Doppler shift: fd = 2 * vr * f0 / c
        doppler_shift = 2 * target_velocity * self.radar_params.frequency / self.c
        
        # Apply relativistic correction if requested and velocity is significant
        if relativistic_correction and abs(target_velocity) > 0.01 * self.c:
            beta = target_velocity / self.c
            gamma = 1 / np.sqrt(1 - beta**2)
            doppler_shift *= gamma * (1 + beta)
        
        # Add atmospheric effects (dispersion)
        if self.environment and hasattr(self.environment, 'atmospheric_dispersion'):
            doppler_shift += self._atmospheric_doppler_effects(target_velocity)
        
        return doppler_shift
    
    def calculate_path_loss(self,
                           range_m: float,
                           frequency: Optional[float] = None,
                           include_atmospheric: bool = True,
                           include_multipath: bool = True) -> float:
        """
        Calculate total path loss including all effects
        
        Args:
            range_m: Propagation range in meters
            frequency: Operating frequency (uses radar frequency if None)
            include_atmospheric: Include atmospheric attenuation
            include_multipath: Include multipath effects
            
        Returns:
            Total path loss in dB
        """
        if frequency is None:
            frequency = self.radar_params.frequency
        
        # Free space path loss: FSPL = (4π*R*f/c)²
        fspl_db = 20 * np.log10(4 * np.pi * range_m * frequency / self.c)
        
        total_loss = fspl_db
        
        # Add atmospheric attenuation
        if include_atmospheric:
            atm_loss = self.environment.atmospheric_attenuation(frequency, range_m)
            total_loss += atm_loss
        
        # Add multipath effects
        if include_multipath and self.propagation_mode != PropagationMode.FREE_SPACE:
            multipath_loss = self._calculate_multipath_loss(range_m, frequency)
            total_loss += multipath_loss
        
        return total_loss
    
    def generate_clutter_returns(self,
                                range_bins: np.ndarray,
                                azimuth_angles: np.ndarray,
                                clutter_params: ClutterParameters,
                                coherent_processing: bool = True) -> np.ndarray:
        """
        Generate realistic clutter returns using physics-based models
        
        Args:
            range_bins: Range bin centers in meters
            azimuth_angles: Azimuth angles in radians
            clutter_params: Clutter modeling parameters
            coherent_processing: Generate coherent (correlated) clutter
            
        Returns:
            2D clutter map (range x azimuth) in linear power units
        """
        num_ranges = len(range_bins)
        num_azimuths = len(azimuth_angles)
        clutter_map = np.zeros((num_ranges, num_azimuths), dtype=complex)
        
        # Generate clutter for each range-azimuth cell
        for i, range_bin in enumerate(range_bins):
            for j, azimuth in enumerate(azimuth_angles):
                
                # Calculate clutter cell area
                range_resolution = self.c / (2 * self.radar_params.bandwidth)
                azimuth_resolution = clutter_params.azimuth_extent
                cell_area = range_bin * azimuth_resolution * range_resolution
                
                # Calculate clutter reflectivity (sigma-0)
                sigma_0 = self._calculate_clutter_reflectivity(
                    clutter_params.clutter_type,
                    range_bin,
                    azimuth,
                    clutter_params.reflectivity_db
                )
                
                # Calculate clutter RCS for this cell
                clutter_rcs = sigma_0 * cell_area
                
                # Apply radar equation for clutter return
                if clutter_rcs > 0 and range_bin > 0:
                    clutter_power = self.radar_equation(
                        range_bin, 
                        clutter_rcs,
                        swerling_model=SwerlingModel.SWERLING_0,  # Clutter doesn't fluctuate like targets
                        include_losses=True
                    )
                    
                    # Generate complex clutter with random phase
                    if coherent_processing:
                        # Coherent clutter with memory
                        clutter_amplitude = np.sqrt(clutter_power)
                        clutter_phase = self._generate_clutter_phase(i, j, clutter_params)
                    else:
                        # Incoherent clutter
                        clutter_amplitude = np.sqrt(clutter_power) * np.random.rayleigh(1.0)
                        clutter_phase = np.random.uniform(0, 2*np.pi)
                    
                    clutter_map[i, j] = clutter_amplitude * np.exp(1j * clutter_phase)
        
        return clutter_map
    
    def calculate_multipath_response(self,
                                   direct_range: float,
                                   target_height: float,
                                   radar_height: float,
                                   multipath_params: Optional[MultipathParameters] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate multipath channel response
        
        Args:
            direct_range: Direct path range in meters
            target_height: Target height in meters
            radar_height: Radar height in meters
            multipath_params: Multipath parameters
            
        Returns:
            Tuple of (path_delays, path_gains) for channel impulse response
        """
        if multipath_params is None:
            multipath_params = MultipathParameters()
        
        if self.propagation_mode == PropagationMode.FREE_SPACE:
            # Only direct path
            return np.array([0.0]), np.array([1.0])
        
        elif self.propagation_mode == PropagationMode.TWO_RAY:
            # Two-ray model: direct + ground reflection
            delays, gains = self._two_ray_model(direct_range, target_height, radar_height)
            
        elif self.propagation_mode == PropagationMode.MULTIPATH:
            # Multiple reflection paths
            delays, gains = self._multipath_model(
                direct_range, target_height, radar_height, multipath_params
            )
            
        else:
            # Default to direct path only
            delays, gains = np.array([0.0]), np.array([1.0])
        
        return delays, gains
    
    def calculate_atmospheric_effects(self,
                                    range_m: float,
                                    elevation_angle: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive atmospheric effects
        
        Args:
            range_m: Propagation range in meters
            elevation_angle: Elevation angle in radians
            
        Returns:
            Dictionary of atmospheric effects
        """
        effects = {}
        
        # Atmospheric attenuation
        atm_loss = self.environment.atmospheric_attenuation(
            self.radar_params.frequency, range_m
        )
        effects['attenuation_db'] = atm_loss
        
        # Refraction effects
        n_surface = self.environment.refraction_index(0)
        n_target = self.environment.refraction_index(range_m * np.sin(elevation_angle))
        effects['refraction_index_surface'] = n_surface
        effects['refraction_index_target'] = n_target
        
        # Ray bending (simplified)
        bending_angle = (n_surface - n_target) * elevation_angle
        effects['ray_bending_rad'] = bending_angle
        
        # Scintillation effects
        scintillation_variance = self._calculate_scintillation(range_m, elevation_angle)
        effects['scintillation_variance'] = scintillation_variance
        
        return effects
    
    def _apply_swerling_fluctuation(self,
                                   base_rcs: float,
                                   swerling_model: SwerlingModel,
                                   target_id: Optional[int] = None) -> float:
        """Apply Swerling fluctuation model to RCS"""
        if not self.enable_fluctuations or swerling_model == SwerlingModel.SWERLING_0:
            return base_rcs
        
        # Use target_id for consistent fluctuations across calls
        if target_id is None:
            target_id = 0
        
        if swerling_model == SwerlingModel.SWERLING_1:
            # Slow fluctuation, exponential distribution
            fluctuation_factor = np.random.exponential(1.0)
            
        elif swerling_model == SwerlingModel.SWERLING_2:
            # Fast fluctuation, exponential distribution
            fluctuation_factor = np.random.exponential(1.0)
            
        elif swerling_model == SwerlingModel.SWERLING_3:
            # Slow fluctuation, chi-squared with 4 DOF (gamma distribution)
            fluctuation_factor = np.random.gamma(2.0, 0.5)
            
        elif swerling_model == SwerlingModel.SWERLING_4:
            # Fast fluctuation, chi-squared with 4 DOF
            fluctuation_factor = np.random.gamma(2.0, 0.5)
            
        else:
            fluctuation_factor = 1.0
        
        return base_rcs * fluctuation_factor
    
    def _calculate_propagation_losses(self,
                                    range_m: float,
                                    target_velocity: Optional[float] = None) -> float:
        """Calculate total propagation losses (linear scale)"""
        # Start with atmospheric losses
        atm_loss_db = self.environment.atmospheric_attenuation(
            self.radar_params.frequency, range_m
        )
        atm_loss_linear = 10**(atm_loss_db / 10)
        
        # Add other propagation effects
        total_loss = atm_loss_linear
        
        # Multipath fading (if enabled)
        if self.propagation_mode != PropagationMode.FREE_SPACE:
            # Simplified multipath loss
            multipath_factor = self._calculate_multipath_factor(range_m)
            total_loss *= multipath_factor
        
        return total_loss
    
    def _calculate_multipath_loss(self, range_m: float, frequency: float) -> float:
        """Calculate multipath loss in dB"""
        if self.propagation_mode == PropagationMode.FREE_SPACE:
            return 0.0
        
        # Simplified multipath loss model
        # Real implementation would use detailed terrain and reflection models
        
        # Two-ray interference pattern
        height_product = 100.0 * 1000.0  # Assume 100m radar, 1000m target height
        critical_distance = 4 * np.sqrt(height_product) / (self.c / frequency)
        
        if range_m > critical_distance:
            # Beyond critical distance, path loss increases as R^4
            additional_loss = 20 * np.log10(range_m / critical_distance)
        else:
            # Within critical distance, oscillatory pattern
            additional_loss = np.random.uniform(-3, 3)  # Simplified
        
        return max(0, additional_loss)
    
    def _calculate_multipath_factor(self, range_m: float) -> float:
        """Calculate multipath fading factor (linear scale)"""
        # Simplified Rayleigh fading model
        if self.propagation_mode == PropagationMode.MULTIPATH:
            # Rayleigh fading for multipath
            fading_amplitude = np.random.rayleigh(1.0)
            return fading_amplitude**2  # Convert amplitude to power
        else:
            return 1.0
    
    def _calculate_clutter_reflectivity(self,
                                      clutter_type: str,
                                      range_m: float,
                                      azimuth: float,
                                      base_reflectivity_db: float) -> float:
        """Calculate clutter reflectivity (sigma-0) in linear units"""
        
        # Base reflectivity
        sigma_0_db = base_reflectivity_db
        
        # Adjust based on clutter type and geometry
        if clutter_type == "land":
            # Land clutter depends on grazing angle
            # Assume flat terrain for simplicity
            radar_height = 10.0  # meters
            grazing_angle = np.arctan(radar_height / range_m)
            sigma_0_db += 10 * np.log10(np.sin(grazing_angle) + 0.001)
            
        elif clutter_type == "sea":
            # Sea clutter depends on sea state and grazing angle
            wind_speed = 10.0  # m/s (moderate sea state)
            grazing_angle = np.arctan(10.0 / range_m)  # Assume 10m radar height
            sigma_0_db += 5 * np.log10(wind_speed) + 20 * np.log10(np.sin(grazing_angle) + 0.001)
            
        elif clutter_type == "urban":
            # Urban clutter has higher reflectivity
            sigma_0_db += 10
            
        elif clutter_type == "weather":
            # Weather clutter (rain/snow)
            if self.environment.conditions.weather == WeatherType.RAIN:
                rain_rate = self.environment.conditions.rain_rate
                sigma_0_db = -30 + 10 * np.log10(rain_rate + 0.1)
            else:
                sigma_0_db = -40  # Light weather
        
        # Add random variations
        sigma_0_db += np.random.normal(0, 3)  # 3 dB standard deviation
        
        # Convert to linear
        sigma_0_linear = 10**(sigma_0_db / 10)
        
        return max(sigma_0_linear, 1e-6)  # Minimum threshold
    
    def _generate_clutter_phase(self,
                              range_idx: int,
                              azimuth_idx: int,
                              clutter_params: ClutterParameters) -> float:
        """Generate clutter phase with temporal correlation"""
        
        # Create unique key for this clutter cell
        cell_key = (range_idx, azimuth_idx)
        
        if cell_key not in self._clutter_memory:
            # Initialize with random phase
            self._clutter_memory[cell_key] = np.random.uniform(0, 2*np.pi)
        
        # Apply temporal correlation
        correlation = clutter_params.temporal_correlation
        old_phase = self._clutter_memory[cell_key]
        
        # Correlated random walk
        phase_increment = np.random.normal(0, np.sqrt(1 - correlation**2) * np.pi/6)
        new_phase = old_phase * correlation + phase_increment
        
        # Wrap phase to [0, 2π]
        new_phase = np.mod(new_phase, 2*np.pi)
        
        # Update memory
        self._clutter_memory[cell_key] = new_phase
        
        return new_phase
    
    def _two_ray_model(self,
                      direct_range: float,
                      target_height: float,
                      radar_height: float) -> Tuple[np.ndarray, np.ndarray]:
        """Two-ray propagation model"""
        
        # Direct path
        direct_delay = direct_range / self.c
        direct_gain = 1.0
        
        # Reflected path
        reflected_range = np.sqrt(direct_range**2 + (target_height + radar_height)**2)
        reflected_delay = reflected_range / self.c
        
        # Reflection coefficient (simplified - depends on polarization and surface)
        reflection_coeff = -0.8  # Typical for ground reflection
        reflected_gain = abs(reflection_coeff) / (reflected_range / direct_range)**2
        
        delays = np.array([direct_delay, reflected_delay])
        gains = np.array([direct_gain, reflected_gain])
        
        return delays, gains
    
    def _multipath_model(self,
                        direct_range: float,
                        target_height: float,
                        radar_height: float,
                        multipath_params: MultipathParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Complex multipath model with multiple reflections"""
        
        if multipath_params.path_delays is not None:
            delays = multipath_params.path_delays
            gains = multipath_params.path_gains
        else:
            # Generate realistic multipath profile
            num_paths = multipath_params.num_paths
            
            # Direct path
            delays = [0.0]
            gains = [1.0]
            
            # Additional paths with random delays and exponentially decaying gains
            for i in range(1, num_paths):
                # Random delay up to several microseconds
                extra_delay = np.random.exponential(1e-6)
                delays.append(extra_delay)
                
                # Exponentially decaying gain
                path_gain = np.exp(-i * 0.5) * np.random.uniform(0.1, 0.8)
                gains.append(path_gain)
            
            delays = np.array(delays)
            gains = np.array(gains)
        
        return delays, gains
    
    def _atmospheric_doppler_effects(self, target_velocity: float) -> float:
        """Calculate atmospheric effects on Doppler shift"""
        # Atmospheric dispersion causes slight frequency-dependent propagation
        # This is typically negligible for radar frequencies but included for completeness
        
        # Simplified model based on atmospheric water vapor
        water_vapor_density = self.environment._calculate_water_vapor_density()
        dispersion_factor = 1e-12 * water_vapor_density  # Very small effect
        
        doppler_shift_correction = dispersion_factor * target_velocity * self.radar_params.frequency
        
        return doppler_shift_correction
    
    def _calculate_scintillation(self, range_m: float, elevation_angle: float) -> float:
        """Calculate atmospheric scintillation variance"""
        # Simplified scintillation model
        # Real model would include Cn² structure parameter, frequency dependence, etc.
        
        # Scintillation is stronger at low elevation angles and long ranges
        if elevation_angle < np.deg2rad(10):  # Below 10 degrees
            scintillation_variance = 0.1 * np.exp(-elevation_angle * 10) * (range_m / 10000)
        else:
            scintillation_variance = 0.01 * (range_m / 100000)
        
        return max(0, min(scintillation_variance, 1.0))
    
    def calculate_signal_to_clutter_ratio(self,
                                        target_rcs: float,
                                        target_range: float,
                                        clutter_params: ClutterParameters,
                                        integration_angle: float = 0.02) -> float:
        """
        Calculate signal-to-clutter ratio for target detection analysis
        
        Args:
            target_rcs: Target RCS in m²
            target_range: Target range in meters
            clutter_params: Clutter parameters
            integration_angle: Radar beamwidth for clutter integration (radians)
            
        Returns:
            Signal-to-clutter ratio in dB
        """
        # Target signal power
        target_power = self.radar_equation(target_range, target_rcs)
        
        # Clutter power calculation
        range_resolution = self.c / (2 * self.radar_params.bandwidth)
        clutter_area = target_range * integration_angle * range_resolution
        
        # Average clutter reflectivity
        avg_sigma_0 = self._calculate_clutter_reflectivity(
            clutter_params.clutter_type, target_range, 0.0, clutter_params.reflectivity_db
        )
        
        clutter_rcs = avg_sigma_0 * clutter_area
        clutter_power = self.radar_equation(target_range, clutter_rcs)
        
        # Signal-to-clutter ratio
        if clutter_power > 0:
            scr_db = 10 * np.log10(target_power / clutter_power)
        else:
            scr_db = 60.0  # Very high SCR if no clutter
        
        return scr_db
    
    def simulate_detection_statistics(self,
                                    target_snr_db: float,
                                    num_pulses: int = 10,
                                    integration_method: str = "coherent",
                                    swerling_model: SwerlingModel = SwerlingModel.SWERLING_1) -> Dict[str, float]:
        """
        Simulate detection statistics for given conditions
        
        Args:
            target_snr_db: Single-pulse target SNR in dB
            num_pulses: Number of pulses for integration
            integration_method: 'coherent' or 'noncoherent'
            swerling_model: Target fluctuation model
            
        Returns:
            Dictionary with detection statistics
        """
        results = {}
        
        # Integration gain
        if integration_method == "coherent":
            integration_gain_db = 10 * np.log10(num_pulses)
        else:  # noncoherent
            integration_gain_db = 10 * np.log10(num_pulses / np.sqrt(2))
        
        # Integrated SNR
        integrated_snr_db = target_snr_db + integration_gain_db
        results['integrated_snr_db'] = integrated_snr_db
        results['integration_gain_db'] = integration_gain_db
        
        # Detection probability (simplified Swerling model)
        snr_linear = 10**(integrated_snr_db / 10)
        
        # Simplified detection probability calculation
        if swerling_model in [SwerlingModel.SWERLING_1, SwerlingModel.SWERLING_2]:
            # Exponential fluctuation
            pfa = 1e-6  # Typical false alarm probability
            threshold = -np.log(pfa)
            pd = 1 - np.exp(-snr_linear / (1 + snr_linear) * threshold)
        else:
            # Chi-squared fluctuation or constant
            pd = 0.5 * erfc(np.sqrt(-np.log(1e-6) - snr_linear))
        
        results['detection_probability'] = max(0, min(pd, 1))
        
        return results