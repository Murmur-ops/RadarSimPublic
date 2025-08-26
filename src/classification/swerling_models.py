#!/usr/bin/env python3
"""
Swerling RCS fluctuation models for realistic radar backscatter simulation
Implements all five Swerling cases for target RCS fluctuation
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class SwerlingCase(Enum):
    """Swerling fluctuation models"""
    SWERLING_0 = 0  # Non-fluctuating (constant RCS)
    SWERLING_1 = 1  # Slow fluctuation, exponential, single dominant scatterer
    SWERLING_2 = 2  # Fast fluctuation, exponential, single dominant scatterer  
    SWERLING_3 = 3  # Slow fluctuation, chi-squared (4 DOF), multiple scatterers
    SWERLING_4 = 4  # Fast fluctuation, chi-squared (4 DOF), multiple scatterers
    SWERLING_5 = 5  # Non-fluctuating (alternate designation)


class SwerlingRCS:
    """
    Implements Swerling RCS fluctuation models for radar targets
    
    Physical interpretation:
    - Swerling 0/5: Non-fluctuating targets (sphere, flat plate normal to radar)
    - Swerling 1/2: Single dominant scatterer plus many small scatterers (aircraft)
    - Swerling 3/4: Many scatterers of comparable size (complex targets)
    
    Slow vs Fast:
    - Slow: RCS constant during dwell (scan-to-scan fluctuation)
    - Fast: RCS changes pulse-to-pulse
    """
    
    def __init__(self, 
                 mean_rcs: float,
                 swerling_case: SwerlingCase = SwerlingCase.SWERLING_1,
                 correlation_time: float = 0.1):
        """
        Initialize Swerling RCS model
        
        Args:
            mean_rcs: Mean RCS value (m²)
            swerling_case: Swerling fluctuation model
            correlation_time: Correlation time for slow fluctuation models (seconds)
        """
        self.mean_rcs = mean_rcs
        self.swerling_case = swerling_case
        self.correlation_time = correlation_time
        
        # State for slow fluctuation models
        self.current_rcs = mean_rcs
        self.last_update_time = 0
        
    def get_rcs(self, 
                time: float,
                pulse_number: Optional[int] = None,
                force_new: bool = False) -> float:
        """
        Get RCS value with appropriate Swerling fluctuation
        
        Args:
            time: Current time (seconds)
            pulse_number: Pulse number for fast fluctuation models
            force_new: Force generation of new RCS value
            
        Returns:
            RCS value with fluctuation applied (m²)
        """
        if self.swerling_case == SwerlingCase.SWERLING_0:
            return self.mean_rcs
        
        # Determine if we need new RCS value
        need_new = force_new
        
        # For slow fluctuation (1, 3), check correlation time
        if self.swerling_case in [SwerlingCase.SWERLING_1, SwerlingCase.SWERLING_3]:
            if time - self.last_update_time > self.correlation_time:
                need_new = True
                self.last_update_time = time
        # For fast fluctuation (2, 4), always generate new
        elif self.swerling_case in [SwerlingCase.SWERLING_2, SwerlingCase.SWERLING_4]:
            need_new = True
        
        if need_new:
            self.current_rcs = self._generate_fluctuated_rcs()
        
        return self.current_rcs
    
    def _generate_fluctuated_rcs(self) -> float:
        """Generate RCS value based on Swerling model"""
        
        if self.swerling_case in [SwerlingCase.SWERLING_1, SwerlingCase.SWERLING_2]:
            # Exponential distribution (Rayleigh amplitude)
            # Single dominant scatterer plus many small ones
            return self.mean_rcs * np.random.exponential(1.0)
        
        elif self.swerling_case in [SwerlingCase.SWERLING_3, SwerlingCase.SWERLING_4]:
            # Chi-squared with 4 degrees of freedom
            # Many scatterers of comparable size
            # This is equivalent to gamma(2, 0.5)
            return self.mean_rcs * np.random.gamma(2, 0.5)
        
        else:
            return self.mean_rcs
    
    def get_detection_probability(self, snr_db: float) -> float:
        """
        Calculate probability of detection for given SNR
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Probability of detection (0-1)
        """
        snr_linear = 10**(snr_db / 10)
        
        if self.swerling_case == SwerlingCase.SWERLING_0:
            # Non-fluctuating target
            # Simplified Marcum Q function approximation
            if snr_db < 10:
                return 0.1
            elif snr_db < 15:
                return 0.5
            else:
                return 0.9
        
        elif self.swerling_case in [SwerlingCase.SWERLING_1, SwerlingCase.SWERLING_2]:
            # Exponential fluctuation
            # Pd = exp(-threshold / (1 + SNR))
            threshold = 10**(13/10)  # ~13 dB threshold
            pd = np.exp(-threshold / (1 + snr_linear))
            return pd
        
        elif self.swerling_case in [SwerlingCase.SWERLING_3, SwerlingCase.SWERLING_4]:
            # Chi-squared fluctuation
            # More complex formula, simplified here
            threshold = 10**(13/10)
            pd = 1 - (1 + 2/snr_linear) * np.exp(-threshold / (1 + snr_linear/2))
            return max(0, min(1, pd))
        
        return 0.5
    
    def get_fluctuation_statistics(self) -> Tuple[float, float]:
        """
        Get statistical properties of RCS fluctuation
        
        Returns:
            (mean, variance) of RCS distribution
        """
        mean = self.mean_rcs
        
        if self.swerling_case in [SwerlingCase.SWERLING_1, SwerlingCase.SWERLING_2]:
            # Exponential: variance = mean²
            variance = self.mean_rcs ** 2
        elif self.swerling_case in [SwerlingCase.SWERLING_3, SwerlingCase.SWERLING_4]:
            # Chi-squared(4): variance = mean²/2
            variance = (self.mean_rcs ** 2) / 2
        else:
            variance = 0
        
        return mean, variance


class BackscatterModel:
    """
    Physical model for radar backscatter including:
    - Swerling RCS fluctuations
    - Aspect angle dependency
    - Polarization effects
    - Frequency dependency
    """
    
    def __init__(self,
                 base_rcs: float,
                 swerling_case: SwerlingCase,
                 target_type: str = "aircraft"):
        """
        Initialize backscatter model
        
        Args:
            base_rcs: Base RCS value (m²)
            swerling_case: Swerling fluctuation model
            target_type: Type of target for aspect modeling
        """
        self.swerling_rcs = SwerlingRCS(base_rcs, swerling_case)
        self.target_type = target_type
        
        # Aspect angle RCS pattern (simplified)
        self.aspect_pattern = self._generate_aspect_pattern()
    
    def _generate_aspect_pattern(self) -> np.ndarray:
        """Generate aspect-dependent RCS pattern"""
        angles = np.linspace(0, 360, 361)
        
        if self.target_type == "aircraft":
            # Aircraft: High RCS from front/rear, low from sides
            pattern = 1 + 0.5 * np.cos(2 * np.radians(angles))
        elif self.target_type == "ship":
            # Ship: High RCS from broadside
            pattern = 1 + 0.7 * np.abs(np.sin(np.radians(angles)))
        elif self.target_type == "missile":
            # Missile: Low RCS from front, higher from sides
            pattern = 0.3 + 0.7 * np.abs(np.sin(np.radians(angles)))
        else:
            # Default: relatively uniform
            pattern = np.ones_like(angles)
        
        return pattern
    
    def get_backscatter(self,
                       time: float,
                       aspect_angle: float,
                       frequency: float = 10e9,
                       polarization: str = "HH") -> complex:
        """
        Calculate complex backscattered field
        
        Args:
            time: Current time (seconds)
            aspect_angle: Aspect angle to target (degrees)
            frequency: Radar frequency (Hz)
            polarization: Polarization (HH, VV, HV, VH)
            
        Returns:
            Complex backscattered field coefficient
        """
        # Get fluctuating RCS
        rcs = self.swerling_rcs.get_rcs(time)
        
        # Apply aspect angle modulation
        aspect_idx = int(aspect_angle % 360)
        rcs *= self.aspect_pattern[aspect_idx]
        
        # Frequency dependency (simplified)
        # Higher frequency = more detailed scattering
        wavelength = 3e8 / frequency
        if self.target_type == "aircraft" and wavelength < 0.1:  # X-band and above
            rcs *= 1.2  # Resonance effects
        
        # Polarization effects
        if polarization == "HH":
            pol_factor = 1.0
        elif polarization == "VV":
            pol_factor = 0.95  # Slightly different for most targets
        elif polarization in ["HV", "VH"]:
            pol_factor = 0.1  # Cross-polarization is much weaker
        else:
            pol_factor = 1.0
        
        rcs *= pol_factor
        
        # Convert RCS to backscatter coefficient
        # σ = 4π |S|² where S is scattering coefficient
        amplitude = np.sqrt(rcs / (4 * np.pi))
        
        # Random phase (target-dependent coherence)
        if self.swerling_rcs.swerling_case in [SwerlingCase.SWERLING_2, SwerlingCase.SWERLING_4]:
            # Fast fluctuation = random phase each pulse
            phase = np.random.uniform(0, 2 * np.pi)
        else:
            # Slow fluctuation = phase changes slowly
            phase = 2 * np.pi * (time * 100) % (2 * np.pi)  # Slowly varying
        
        return amplitude * np.exp(1j * phase)
    
    def get_doppler_spectrum(self, 
                            velocity: float,
                            frequency: float = 10e9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Doppler spectrum of backscattered signal
        
        Args:
            velocity: Radial velocity (m/s)
            frequency: Radar frequency (Hz)
            
        Returns:
            (frequencies, spectrum) arrays
        """
        wavelength = 3e8 / frequency
        doppler_shift = 2 * velocity / wavelength
        
        # Create spectrum centered on Doppler shift
        freqs = np.linspace(doppler_shift - 100, doppler_shift + 100, 201)
        
        # Spectrum shape depends on target type
        if self.target_type == "aircraft":
            # Narrow main peak with some spreading
            spectrum = np.exp(-(freqs - doppler_shift)**2 / (2 * 10**2))
        elif self.target_type == "ship":
            # Broader due to sea motion
            spectrum = np.exp(-(freqs - doppler_shift)**2 / (2 * 30**2))
        else:
            # Default narrow spectrum
            spectrum = np.exp(-(freqs - doppler_shift)**2 / (2 * 5**2))
        
        return freqs, spectrum


def demonstrate_swerling_models():
    """Demonstrate different Swerling models"""
    import matplotlib.pyplot as plt
    
    mean_rcs = 10.0  # 10 m² mean RCS
    time_samples = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Swerling RCS Fluctuation Models', fontsize=14, fontweight='bold')
    
    models = [
        (SwerlingCase.SWERLING_0, "Swerling 0\n(Non-fluctuating)"),
        (SwerlingCase.SWERLING_1, "Swerling 1\n(Slow, Exponential)"),
        (SwerlingCase.SWERLING_2, "Swerling 2\n(Fast, Exponential)"),
        (SwerlingCase.SWERLING_3, "Swerling 3\n(Slow, Chi-squared)"),
        (SwerlingCase.SWERLING_4, "Swerling 4\n(Fast, Chi-squared)"),
    ]
    
    for idx, (swerling_case, title) in enumerate(models):
        ax = axes[idx // 3, idx % 3]
        
        model = SwerlingRCS(mean_rcs, swerling_case, correlation_time=0.1)
        rcs_values = [model.get_rcs(t) for t in time_samples]
        
        ax.plot(time_samples, rcs_values, 'b-', linewidth=0.5)
        ax.axhline(y=mean_rcs, color='r', linestyle='--', label='Mean RCS')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RCS (m²)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(30, max(rcs_values))])
        ax.legend(loc='upper right')
    
    # Hide the 6th subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('swerling_models_demo.png', dpi=150)
    print("Saved Swerling models demonstration to swerling_models_demo.png")
    
    return fig


if __name__ == "__main__":
    demonstrate_swerling_models()