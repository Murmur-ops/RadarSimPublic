"""
Core radar model implementation
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Import physical constants
try:
    from .constants import SPEED_OF_LIGHT, BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE
except ImportError:
    # Fallback for backward compatibility
    SPEED_OF_LIGHT = 3e8
    BOLTZMANN_CONSTANT = 1.38e-23
    STANDARD_TEMPERATURE = 290


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
        return SPEED_OF_LIGHT / self.frequency
    
    @property
    def max_unambiguous_range(self) -> float:
        """Calculate maximum unambiguous range"""
        return SPEED_OF_LIGHT / (2 * self.prf)
    
    @property
    def range_resolution(self) -> float:
        """Calculate range resolution"""
        return SPEED_OF_LIGHT / (2 * self.bandwidth)


class Radar:
    """Main radar simulation class"""
    
    def __init__(self, params: RadarParameters):
        self.params = params
        self.k_boltzmann = BOLTZMANN_CONSTANT
        self.temperature = STANDARD_TEMPERATURE
        
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
    
    def detection_probability(self, snr_db: float, pfa: float = 1e-6, 
                            swerling_model: int = 1, n_pulses: int = 1) -> float:
        """
        Calculate probability of detection using Swerling models
        
        Args:
            snr_db: Signal-to-Noise Ratio in dB
            pfa: Probability of false alarm
            swerling_model: Swerling case (0, 1, 2, 3, 4)
            n_pulses: Number of integrated pulses
            
        Returns:
            Probability of detection
        """
        from scipy import special, stats
        
        snr_linear = 10**(snr_db / 10)
        
        # Calculate detection threshold from false alarm rate
        # For non-coherent integration of n_pulses
        threshold_factor = stats.chi2.ppf(1 - pfa, 2 * n_pulses) / (2 * n_pulses)
        
        if swerling_model == 0:
            # Swerling 0: Non-fluctuating (constant RCS)
            # Use Marcum Q-function for Rician distribution
            from scipy.special import marcumq
            a = np.sqrt(2 * n_pulses * snr_linear)
            b = np.sqrt(2 * n_pulses * threshold_factor)
            pd = float(marcumq(n_pulses, a, b))
            
        elif swerling_model == 1:
            # Swerling 1: Slow fluctuation, chi-squared with 2 DOF
            # RCS constant during pulse train, varies scan-to-scan
            pd = (1 + threshold_factor / snr_linear) ** (-n_pulses)
            
        elif swerling_model == 2:
            # Swerling 2: Fast fluctuation, chi-squared with 2 DOF
            # RCS varies pulse-to-pulse
            if n_pulses == 1:
                pd = np.exp(-threshold_factor / snr_linear)
            else:
                # For multiple pulses with independent fluctuation
                pd = special.gammaincc(n_pulses, n_pulses * threshold_factor / snr_linear)
            
        elif swerling_model == 3:
            # Swerling 3: Slow fluctuation, chi-squared with 4 DOF
            # RCS constant during pulse train
            term1 = 1 + threshold_factor / (snr_linear / 2)
            term2 = 1 - threshold_factor / (n_pulses * (2 + snr_linear))
            pd = term1 ** (-n_pulses) * term2
            
        elif swerling_model == 4:
            # Swerling 4: Fast fluctuation, chi-squared with 4 DOF
            # RCS varies pulse-to-pulse
            if n_pulses == 1:
                a = threshold_factor / snr_linear
                pd = np.exp(-a/2) * (1 + a/2)
            else:
                # For multiple pulses
                sum_term = 0
                for k in range(n_pulses):
                    sum_term += special.binom(n_pulses - 1 + k, k) * \
                               (2 / (2 + snr_linear)) ** k
                pd = ((2 + snr_linear) / 2) ** (-n_pulses) * \
                     np.exp(-n_pulses * threshold_factor / (2 + snr_linear)) * sum_term
        else:
            raise ValueError(f"Invalid Swerling model: {swerling_model}. Must be 0, 1, 2, 3, or 4")
        
        # Ensure Pd is in valid range
        pd = np.clip(pd, 0.0, 1.0)
        
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