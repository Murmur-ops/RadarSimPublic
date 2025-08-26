"""
Core radar model implementation
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RadarParameters:
    """Radar system parameters"""
    frequency: float  # Hz
    power: float  # Watts
    antenna_gain: float  # dB
    pulse_width: float  # seconds
    prf: float  # Pulse Repetition Frequency (Hz)
    bandwidth: float  # Hz
    noise_figure: float  # dB
    losses: float  # dB
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength from frequency"""
        c = 3e8  # Speed of light
        return c / self.frequency
    
    @property
    def max_unambiguous_range(self) -> float:
        """Calculate maximum unambiguous range"""
        c = 3e8
        return c / (2 * self.prf)
    
    @property
    def range_resolution(self) -> float:
        """Calculate range resolution"""
        c = 3e8
        return c / (2 * self.bandwidth)


class Radar:
    """Main radar simulation class"""
    
    def __init__(self, params: RadarParameters):
        self.params = params
        self.k_boltzmann = 1.38e-23  # Boltzmann constant
        self.temperature = 290  # Kelvin
        
    def radar_equation(self, range_m: float, rcs: float) -> float:
        """
        Calculate received power using radar equation
        
        Args:
            range_m: Range to target in meters
            rcs: Radar Cross Section in m^2
            
        Returns:
            Received power in Watts
        """
        pt = self.params.power
        gt = 10**(self.params.antenna_gain / 10)
        gr = gt  # Assuming monostatic radar
        lambda_val = self.params.wavelength
        sigma = rcs
        losses = 10**(self.params.losses / 10)
        
        numerator = pt * gt * gr * lambda_val**2 * sigma
        denominator = (4 * np.pi)**3 * range_m**4 * losses
        
        return numerator / denominator
    
    def snr(self, range_m: float, rcs: float) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            range_m: Range to target in meters
            rcs: Radar Cross Section in m^2
            
        Returns:
            SNR in dB
        """
        pr = self.radar_equation(range_m, rcs)
        
        # Calculate noise power
        nf = 10**(self.params.noise_figure / 10)
        noise_power = self.k_boltzmann * self.temperature * self.params.bandwidth * nf
        
        snr_linear = pr / noise_power
        return 10 * np.log10(snr_linear)
    
    def detection_probability(self, snr_db: float, pfa: float = 1e-6) -> float:
        """
        Calculate probability of detection using simplified model
        
        Args:
            snr_db: Signal-to-Noise Ratio in dB
            pfa: Probability of false alarm
            
        Returns:
            Probability of detection
        """
        from scipy import special
        
        snr_linear = 10**(snr_db / 10)
        
        # Simplified Swerling 1 model
        threshold = -np.log(pfa)
        pd = np.exp(-threshold / (1 + snr_linear))
        
        return pd
    
    def generate_pulse(self, samples: int = 1000) -> np.ndarray:
        """
        Generate a radar pulse
        
        Args:
            samples: Number of samples in the pulse
            
        Returns:
            Complex pulse waveform
        """
        t = np.linspace(0, self.params.pulse_width, samples)
        
        # Simple rectangular pulse with carrier
        carrier = np.exp(1j * 2 * np.pi * self.params.frequency * t)
        envelope = np.ones_like(t)
        
        return envelope * carrier
    
    def range_profile(self, targets: List[Tuple[float, float]], 
                     max_range: float = 10000,
                     num_bins: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate range profile for multiple targets
        
        Args:
            targets: List of (range, rcs) tuples
            max_range: Maximum range to display
            num_bins: Number of range bins
            
        Returns:
            Range bins and power profile
        """
        ranges = np.linspace(0, max_range, num_bins)
        profile = np.zeros(num_bins)
        
        c = 3e8
        range_bin_size = max_range / num_bins
        
        for target_range, target_rcs in targets:
            if target_range <= max_range:
                # Find closest range bin
                bin_idx = int(target_range / range_bin_size)
                if bin_idx < num_bins:
                    # Calculate received power for this target
                    power = self.radar_equation(target_range, target_rcs)
                    profile[bin_idx] += power
        
        # Add noise
        noise_power = self.k_boltzmann * self.temperature * self.params.bandwidth * \
                     10**(self.params.noise_figure / 10)
        noise = np.random.normal(0, np.sqrt(noise_power), num_bins)
        profile += np.abs(noise)
        
        # Convert to dB
        profile_db = 10 * np.log10(profile + 1e-20)  # Add small value to avoid log(0)
        
        return ranges, profile_db
    
    def doppler_shift(self, velocity: float) -> float:
        """
        Calculate Doppler shift for a moving target
        
        Args:
            velocity: Radial velocity in m/s (positive = approaching)
            
        Returns:
            Doppler shift in Hz
        """
        return 2 * velocity * self.params.frequency / 3e8
    
    def max_unambiguous_velocity(self) -> float:
        """
        Calculate maximum unambiguous velocity
        
        Returns:
            Maximum velocity in m/s
        """
        return self.params.wavelength * self.params.prf / 4